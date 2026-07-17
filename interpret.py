"""
interpret.py — Explain RETAIN predictions (per-code contributions)

This is the modern replacement for ../test_retain.py.

For each patient it writes:
  - every visit
  - every medical code in that visit
  - a contribution score (positive = pushed risk UP, negative = pushed risk DOWN)
  - the model's predicted probability and the true label

Contribution formula (same as the original):
  contrib(i, code) = w_output · (alpha_i * beta_i * W_emb[code])

Example:

  python interpret.py model.3.pt visits.pkl labels.pkl types.pkl explanations.txt
"""

# argparse reads command-line arguments.
import argparse

# os for path helpers.
import os

# NumPy for small numeric conversions.
import numpy as np

# PyTorch for loading the model and running it.
import torch

# RETAIN model class.
from model import RETAIN

# Data helpers: load pickles, pad one patient, convert to tensors.
from data_utils import load_test_data, load_pickle, pad_batch, batch_to_tensors


def load_checkpoint(model_file, device):
    """
    Load a .pt file saved by train.py and rebuild the RETAIN model.
    """
    # map_location ensures the weights load even if they were saved on a GPU
    # and we are running on CPU (or the other way around).
    ckpt = torch.load(model_file, map_location=device)

    # Recreate the model using the settings stored in the checkpoint.
    model = RETAIN(
        input_dim_size=ckpt["input_dim_size"],
        emb_dim_size=ckpt["emb_dim_size"],
        alpha_hidden_size=ckpt["alpha_hidden_size"],
        beta_hidden_size=ckpt["beta_hidden_size"],
        use_time=ckpt["use_time"],
        emb_finetune=ckpt.get("emb_finetune", True),
        # At test time we do not want dropout, so keep_prob=1.0 → dropout p=0.
        keep_prob_emb=1.0,
        keep_prob_context=1.0,
        pretrained_embedding=None,
    ).to(device)

    # Copy the trained weights into the new model object.
    model.load_state_dict(ckpt["model_state_dict"])

    # Evaluation mode: disables dropout.
    model.eval()

    return model, ckpt


def interpret_patients(args):
    """
    Main interpretation loop over every patient in the visit file.
    """
    # Pick CPU or GPU.
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model + training settings from the checkpoint.
    print("Loading the parameters ...")
    model, ckpt = load_checkpoint(args.model_file, device)

    # Read visit / label / optional time data.
    print("Loading data ...")
    seqs, labels, times = load_test_data(args.seq_file, args.label_file, args.time_file)
    print("done")

    # Mapping: string medical code → integer id (from process_mimic .types file).
    types = load_pickle(args.type_file)
    # Reverse mapping: integer id → string name (for readable output).
    rtypes = {v: k for k, v in types.items()}

    # Whether this model was trained with time features.
    use_time = bool(ckpt["use_time"])
    use_log_time = bool(ckpt.get("use_log_time", True))
    input_dim_size = ckpt["input_dim_size"]

    # Make sure the folder for the output file exists.
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Contribution calculation start!!")

    # Open the output text file for writing.
    with open(args.out_file, "w", encoding="utf-8") as outfd:
        # Loop over patients one by one (same as the original script).
        for index in range(len(seqs)):
            # Progress message every 100 patients.
            if index % 100 == 0:
                print(f"processed {index} patients")

            # This patient's visit list and true label.
            patient = seqs[index]
            label = labels[index]

            # Build a batch of size 1 for padding helpers.
            batch_seqs = [patient]
            batch_times = None if (not use_time or times is None) else [times[index]]

            # Pad into arrays.
            x, t, lengths = pad_batch(
                batch_seqs,
                input_dim_size,
                times=batch_times,
                use_log_time=use_log_time,
            )

            # Convert to tensors.
            x_t, lengths_t, _, t_t = batch_to_tensors(x, lengths, t=t, device=device)

            # Forward pass WITH attention values returned.
            with torch.no_grad():
                y_hat, alpha, beta, emb, mask = model(
                    x_t, lengths_t, times=t_t, return_attention=True
                )

            # Squeeze batch dimension: we only have 1 patient.
            # alpha_p shape: (time,), beta_p shape: (time, emb_dim)
            alpha_p = alpha[0]
            beta_p = beta[0]
            score = float(y_hat[0].cpu().item())
            n_visits = int(lengths[0])

            # Build a human-readable text block for this patient.
            buf = ""
            for i in range(n_visits):
                # Header for this visit.
                buf += f"-------------- visit_index:{i} ---------------\n"

                # Each medical code present in this visit.
                visit = patient[i]
                for code in visit:
                    # Contribution of this code at this visit (scalar).
                    contrib = model.contribution_for_code(alpha_p[i], beta_p[i], int(code))
                    # Look up the readable name; fall back to the integer if missing.
                    name = rtypes.get(int(code), str(code))
                    buf += f"{name}:{float(contrib.cpu().item()):f}  "

                buf += "\n------------------------------------\n"

            # Patient-level summary line.
            buf += f"patient_index:{index}, label:{int(label)}, score:{score:f}\n\n"

            # Write this patient to the output file.
            outfd.write(buf + "\n")

    print(f"Wrote contributions to {args.out_file}")


def parse_arguments():
    """
    Command-line interface mirroring test_retain.py.
    """
    parser = argparse.ArgumentParser(description="Interpret a trained RETAIN (PyTorch) model")

    parser.add_argument("model_file", type=str, help="Path to .pt checkpoint from train.py")
    parser.add_argument("seq_file", type=str, help="Pickled visit file")
    parser.add_argument("label_file", type=str, help="Pickled label file")
    parser.add_argument("type_file", type=str, help="Pickled dict mapping code string → int")
    parser.add_argument("out_file", type=str, help="Where to write contribution text")

    parser.add_argument("--time_file", type=str, default="", help="Optional pickled time file")
    parser.add_argument(
        "--use_log_time",
        type=int,
        default=1,
        choices=[0, 1],
        help="Kept for CLI compatibility; checkpoint's use_log_time is used",
    )
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse flags and run interpretation.
    args = parse_arguments()
    interpret_patients(args)
