"""
process_mimic.py — Prepare MIMIC-III CSVs for RETAIN (Python 3)

This is a modernized port of ../process_mimic.py.

What it does:
  1. Read PATIENTS.csv → who died in hospital (mortality label)
  2. Read ADMISSIONS.csv → which admissions each patient had + dates
  3. Read DIAGNOSES_ICD.csv → ICD diagnosis codes per admission
  4. Keep only patients with at least 2 visits
  5. Convert string ICD codes into integer ids
  6. Save several pickle files that train.py / interpret.py can use

Usage (put this script next to the MIMIC CSV files, or pass full paths):

  python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv output_prefix
"""

# sys lets us read command-line arguments as a simple list.
import sys

# pickle saves Python objects (lists, dicts) to disk.
import pickle

# datetime parses admission timestamps from the CSV.
from datetime import datetime


def convert_to_icd9(dx_str):
    """
    Insert a dot into an ICD-9 code string in the usual clinical format.

    Examples:
      '25000' → '250.00'
      'E8497' → 'E849.7'  (E-codes are external-cause codes)
    """
    # E-codes: keep letter + 3 digits, then the rest after a dot.
    if dx_str.startswith("E"):
        if len(dx_str) > 4:
            return dx_str[:4] + "." + dx_str[4:]
        return dx_str

    # Regular codes: first 3 digits, then the rest after a dot.
    if len(dx_str) > 3:
        return dx_str[:3] + "." + dx_str[3:]
    return dx_str


def convert_to_3digit_icd9(dx_str):
    """
    Keep only the 3-digit (or E + 3-digit) category of an ICD-9 code.

    Using categories makes the vocabulary smaller and often more interpretable.
    """
    if dx_str.startswith("E"):
        if len(dx_str) > 4:
            return dx_str[:4]
        return dx_str

    if len(dx_str) > 3:
        return dx_str[:3]
    return dx_str


def main():
    """
    Run the full MIMIC → RETAIN preprocessing pipeline.
    """
    # Expect exactly 4 arguments after the script name.
    if len(sys.argv) != 5:
        print(
            "Usage: python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv "
            "PATIENTS.csv <output_prefix>"
        )
        sys.exit(1)

    # Paths from the command line.
    admission_file = sys.argv[1]
    diagnosis_file = sys.argv[2]
    patients_file = sys.argv[3]
    out_file = sys.argv[4]

    # ------------------------------------------------------------------
    # Step 1: mortality labels from PATIENTS.csv
    # ------------------------------------------------------------------
    print("Collecting mortality information")

    # Dictionary: patient_id → 1 if died in hospital, else 0.
    pid_dod_map = {}

    # Open the patients file and skip the header line.
    with open(patients_file, "r", encoding="utf-8") as infd:
        # Discard the CSV header.
        infd.readline()
        for line in infd:
            # Split the CSV row into columns.
            tokens = line.strip().split(",")
            # SUBJECT_ID is column index 1 in the MIMIC PATIENTS table layout
            # used by the original script.
            pid = int(tokens[1])
            # DOD_HOSP (death date in hospital) is column index 5.
            dod_hosp = tokens[5]
            # If that field is non-empty, the patient died in hospital.
            pid_dod_map[pid] = 1 if len(dod_hosp) > 0 else 0

    # ------------------------------------------------------------------
    # Step 2: map patients → admissions, and admissions → dates
    # ------------------------------------------------------------------
    print("Building pid-admission mapping, admission-date mapping")

    # patient_id → list of admission ids
    pid_adm_map = {}
    # admission_id → datetime of admission
    adm_date_map = {}

    with open(admission_file, "r", encoding="utf-8") as infd:
        infd.readline()
        for line in infd:
            tokens = line.strip().split(",")
            pid = int(tokens[1])
            adm_id = int(tokens[2])
            # Parse "YYYY-MM-DD HH:MM:SS" into a datetime object.
            adm_time = datetime.strptime(tokens[3], "%Y-%m-%d %H:%M:%S")
            adm_date_map[adm_id] = adm_time
            # Append this admission to the patient's list (create list if needed).
            if pid in pid_adm_map:
                pid_adm_map[pid].append(adm_id)
            else:
                pid_adm_map[pid] = [adm_id]

    # ------------------------------------------------------------------
    # Step 3: map admissions → lists of diagnosis code strings
    # ------------------------------------------------------------------
    print("Building admission-dxList mapping")

    # Full ICD-9 string codes per admission.
    adm_dx_map = {}
    # 3-digit category codes per admission.
    adm_dx_map_3digit = {}

    with open(diagnosis_file, "r", encoding="utf-8") as infd:
        infd.readline()
        for line in infd:
            tokens = line.strip().split(",")
            adm_id = int(tokens[2])
            # tokens[4] is often quoted like "25000"; [1:-1] strips the quotes.
            raw_code = tokens[4][1:-1]
            # Prefix with 'D_' so diagnosis codes are clearly labeled as strings.
            dx_str = "D_" + convert_to_icd9(raw_code)
            dx_str_3digit = "D_" + convert_to_3digit_icd9(raw_code)

            # Append to full-code list.
            if adm_id in adm_dx_map:
                adm_dx_map[adm_id].append(dx_str)
            else:
                adm_dx_map[adm_id] = [dx_str]

            # Append to 3-digit list.
            if adm_id in adm_dx_map_3digit:
                adm_dx_map_3digit[adm_id].append(dx_str_3digit)
            else:
                adm_dx_map_3digit[adm_id] = [dx_str_3digit]

    # ------------------------------------------------------------------
    # Step 4: build chronological visit sequences (patients with ≥ 2 visits)
    # ------------------------------------------------------------------
    print("Building pid-sortedVisits mapping")

    # patient_id → list of (date, code_list) sorted by date
    pid_seq_map = {}
    pid_seq_map_3digit = {}

    # .items() is the Python 3 replacement for Python 2's .iteritems().
    for pid, adm_id_list in pid_adm_map.items():
        # RETAIN needs longitudinal data: skip single-visit patients.
        if len(adm_id_list) < 2:
            continue

        # Pair each admission with its date and diagnosis list, then sort by date.
        sorted_list = sorted(
            [(adm_date_map[adm_id], adm_dx_map[adm_id]) for adm_id in adm_id_list]
        )
        pid_seq_map[pid] = sorted_list

        sorted_list_3digit = sorted(
            [
                (adm_date_map[adm_id], adm_dx_map_3digit[adm_id])
                for adm_id in adm_id_list
            ]
        )
        pid_seq_map_3digit[pid] = sorted_list_3digit

    # ------------------------------------------------------------------
    # Step 5: flatten into parallel lists (pids, dates, seqs, morts)
    # ------------------------------------------------------------------
    print("Building pids, dates, mortality_labels, strSeqs")

    pids = []
    dates = []
    seqs = []
    morts = []

    for pid, visits in pid_seq_map.items():
        pids.append(pid)
        morts.append(pid_dod_map[pid])

        seq = []
        date = []
        for visit in visits:
            # visit[0] = datetime, visit[1] = list of string codes
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)

    print("Building pids, dates, strSeqs for 3digit ICD9 code")
    seqs_3digit = []
    for pid, visits in pid_seq_map_3digit.items():
        seq = []
        for visit in visits:
            seq.append(visit[1])
        seqs_3digit.append(seq)

    # ------------------------------------------------------------------
    # Step 6: convert string codes → integer ids (+ build vocab dict)
    # ------------------------------------------------------------------
    print("Converting strSeqs to intSeqs, and making types")

    # types maps string code → integer id
    types = {}
    new_seqs = []

    for patient in seqs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in visit:
                # Reuse existing id, or assign the next available integer.
                if code in types:
                    new_visit.append(types[code])
                else:
                    types[code] = len(types)
                    new_visit.append(types[code])
            new_patient.append(new_visit)
        new_seqs.append(new_patient)

    print("Converting strSeqs to intSeqs, and making types for 3digit ICD9 code")
    types_3digit = {}
    new_seqs_3digit = []

    for patient in seqs_3digit:
        new_patient = []
        for visit in patient:
            new_visit = []
            # set(visit) removes duplicate codes within the same visit.
            for code in set(visit):
                if code in types_3digit:
                    new_visit.append(types_3digit[code])
                else:
                    types_3digit[code] = len(types_3digit)
                    new_visit.append(types_3digit[code])
            new_patient.append(new_visit)
        new_seqs_3digit.append(new_patient)

    # ------------------------------------------------------------------
    # Step 7: save everything as pickle files
    # ------------------------------------------------------------------
    # protocol=-1 means "use the newest binary protocol available".
    pickle.dump(pids, open(out_file + ".pids", "wb"), -1)
    pickle.dump(dates, open(out_file + ".dates", "wb"), -1)
    pickle.dump(morts, open(out_file + ".morts", "wb"), -1)
    pickle.dump(new_seqs, open(out_file + ".seqs", "wb"), -1)
    pickle.dump(types, open(out_file + ".types", "wb"), -1)
    pickle.dump(new_seqs_3digit, open(out_file + ".3digitICD9.seqs", "wb"), -1)
    pickle.dump(types_3digit, open(out_file + ".3digitICD9.types", "wb"), -1)

    print(f"Wrote outputs with prefix: {out_file}")
    print(f"  patients: {len(pids)}")
    print(f"  unique full ICD codes: {len(types)}")
    print(f"  unique 3-digit ICD codes: {len(types_3digit)}")


if __name__ == "__main__":
    # Only run main() when this file is executed directly.
    main()
