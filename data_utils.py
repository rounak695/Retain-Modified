"""
data_utils.py — Load patient visit data and turn it into model-ready batches

The original Theano code used Python 2's cPickle and time-major arrays
(shape: time, batch, features). Here we use:

  - Python 3's pickle module
  - batch-first tensors (shape: batch, time, features) which is normal in PyTorch

The DATA FORMAT is unchanged from the original README:

  Visit file: list of patients → each patient is a list of visits →
              each visit is a list of integer medical codes.

  Label file: list of 0/1 labels, one per patient.

  Time file (optional): list of patients → list of time numbers per visit.
"""

# pickle = Python's tool to save/load Python objects to a file.
import pickle

# NumPy = library for fast numeric arrays (we use it before converting to torch).
import numpy as np

# torch = PyTorch; we convert NumPy batches into torch tensors for the model.
import torch


# Fractions used when --simple_load splits one big file into train/valid/test.
# These match the original retain.py constants.
_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1


def load_pickle(path):
    """
    Open a pickle file and return the Python object inside.

    'rb' means read-binary (pickle files are binary, not plain text).
    """
    # with open(...) automatically closes the file when we leave this block.
    with open(path, "rb") as f:
        # pickle.load reads the object (list, dict, etc.) from the file.
        return pickle.load(f)


def sort_by_length(seqs, labels, times=None):
    """
    Sort patients from shortest history to longest.

    Why? Mini-batches of similar lengths waste less space on padding.
    This mirrors len_argsort() in the original code.
    """
    # For each patient index i, key = how many visits that patient has.
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))

    # Reorder sequences and labels using that order.
    seqs = [seqs[i] for i in order]
    labels = [labels[i] for i in order]

    # Reorder times the same way if they exist.
    if times is not None:
        times = [times[i] for i in order]

    return seqs, labels, times


def load_data(seq_file, label_file, time_file=""):
    """
    Load already-split datasets: *.train, *.valid, *.test

    Example: seq_file='my_visits' loads:
      my_visits.train, my_visits.valid, my_visits.test
    """
    # Load visit sequences for each split.
    train_x = load_pickle(seq_file + ".train")
    valid_x = load_pickle(seq_file + ".valid")
    test_x = load_pickle(seq_file + ".test")

    # Load labels for each split.
    train_y = load_pickle(label_file + ".train")
    valid_y = load_pickle(label_file + ".valid")
    test_y = load_pickle(label_file + ".test")

    # Times are optional; default to None for each split.
    train_t = valid_t = test_t = None

    # If the user passed a time file prefix, load the three splits.
    if len(time_file) > 0:
        train_t = load_pickle(time_file + ".train")
        valid_t = load_pickle(time_file + ".valid")
        test_t = load_pickle(time_file + ".test")

    # Sort each split by sequence length (same as original).
    train_x, train_y, train_t = sort_by_length(train_x, train_y, train_t)
    valid_x, valid_y, valid_t = sort_by_length(valid_x, valid_y, valid_t)
    test_x, test_y, test_t = sort_by_length(test_x, test_y, test_t)

    # Return three tuples: (visits, labels, times)
    return (train_x, train_y, train_t), (valid_x, valid_y, valid_t), (test_x, test_y, test_t)


def load_data_simple(seq_file, label_file, time_file=""):
    """
    Load ONE visit file + ONE label file, then randomly split into
    train / valid / test (same ratios as original --simple_load).
    """
    # Load the full lists from disk.
    sequences = np.array(load_pickle(seq_file), dtype=object)
    labels = np.array(load_pickle(label_file))

    # Load times only if a path was provided.
    times = None
    if len(time_file) > 0:
        times = np.array(load_pickle(time_file), dtype=object)

    # How many patients are there?
    data_size = len(labels)

    # Fixed seed so the split is reproducible (same as original np.random.seed(0)).
    np.random.seed(0)

    # Random permutation of patient indices: [3, 0, 5, 1, ...]
    ind = np.random.permutation(data_size)

    # How many patients go to test and validation.
    n_test = int(_TEST_RATIO * data_size)
    n_valid = int(_VALIDATION_RATIO * data_size)

    # Slice the shuffled indices into three groups.
    test_indices = ind[:n_test]
    valid_indices = ind[n_test : n_test + n_valid]
    train_indices = ind[n_test + n_valid :]

    # Gather visits and labels for each split.
    train_x, train_y = sequences[train_indices], labels[train_indices]
    valid_x, valid_y = sequences[valid_indices], labels[valid_indices]
    test_x, test_y = sequences[test_indices], labels[test_indices]

    # Gather times if present.
    train_t = valid_t = test_t = None
    if times is not None:
        train_t = times[train_indices]
        valid_t = times[valid_indices]
        test_t = times[test_indices]

    # Convert NumPy object arrays back to plain Python lists, then sort.
    train_x, train_y, train_t = sort_by_length(list(train_x), list(train_y), None if train_t is None else list(train_t))
    valid_x, valid_y, valid_t = sort_by_length(list(valid_x), list(valid_y), None if valid_t is None else list(valid_t))
    test_x, test_y, test_t = sort_by_length(list(test_x), list(test_y), None if test_t is None else list(test_t))

    return (train_x, train_y, train_t), (valid_x, valid_y, valid_t), (test_x, test_y, test_t)


def load_test_data(seq_file, label_file, time_file=""):
    """
    Load a single visit/label file for interpretation (no train/valid/test suffix).
    Matches test_retain.py's load_data().
    """
    # Read visits and labels.
    seqs = load_pickle(seq_file)
    labels = load_pickle(label_file)

    # Optional times.
    times = None
    if len(time_file) > 0:
        times = load_pickle(time_file)

    # Sort by length for consistent ordering.
    seqs, labels, times = sort_by_length(list(seqs), list(labels), None if times is None else list(times))

    return seqs, labels, times


def pad_batch(seqs, input_dim_size, times=None, use_log_time=True):
    """
    Convert a list of patients into padded NumPy arrays for one mini-batch.

    This replaces padMatrixWithTime / padMatrixWithoutTime from the original,
    but uses batch-first layout:

      x shape: (batch, max_time, input_dim_size)
      t shape: (batch, max_time)   — only if times is not None
      lengths: (batch,)

    Each visit becomes a multi-hot vector:
      codes [1, 4] → vector with 1.0 at positions 1 and 4, else 0.0
    """
    # Number of patients in this batch.
    batch_size = len(seqs)

    # Real number of visits for each patient.
    lengths = np.array([len(seq) for seq in seqs], dtype=np.int64)

    # Longest history in this batch (all shorter ones will be padded).
    max_len = int(lengths.max()) if batch_size > 0 else 0

    # Allocate zeros: padding visits stay all-zero multi-hot vectors.
    x = np.zeros((batch_size, max_len, input_dim_size), dtype=np.float32)

    # Optional time matrix (also padded with zeros).
    t = None
    if times is not None:
        t = np.zeros((batch_size, max_len), dtype=np.float32)

    # Fill in each patient's real visits.
    for i, seq in enumerate(seqs):
        for j, visit_codes in enumerate(seq):
            # visit_codes is a list of integer code ids present in this visit.
            # Setting those indices to 1.0 creates the multi-hot vector.
            # (Guard against empty visits.)
            if len(visit_codes) > 0:
                x[i, j, visit_codes] = 1.0

        # Copy time values for this patient's real visits.
        if times is not None:
            t[i, : lengths[i]] = times[i]

    # Optional log(time + 1) to reduce the effect of huge time gaps.
    if t is not None and use_log_time:
        t = np.log(t + 1.0)

    return x, t, lengths


def batch_to_tensors(x, lengths, y=None, t=None, device="cpu"):
    """
    Convert NumPy arrays from pad_batch into PyTorch tensors on the right device.

    device is usually 'cpu' or 'cuda' (GPU).
    """
    # Float tensor of multi-hot visits.
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)

    # Integer tensor of lengths.
    lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device)

    # Labels are optional (not needed when only predicting).
    y_t = None
    if y is not None:
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)

    # Times are optional.
    t_t = None
    if t is not None:
        t_t = torch.as_tensor(t, dtype=torch.float32, device=device)

    return x_t, lengths_t, y_t, t_t


def iter_minibatches(dataset, batch_size, shuffle=False):
    """
    Yield slices of a dataset as mini-batches.

    dataset = (seqs, labels, times) where times may be None.

    When shuffle=True, batch order matches the original retain.py loop:
      for index in random.sample(range(n_batches), n_batches):
          batch = data[index*batchSize : (index+1)*batchSize]
    """
    # Python's random module (same as original), not NumPy.
    import random

    # Unpack the three parts of the dataset tuple.
    seqs, labels, times = dataset

    # How many patients total.
    n = len(seqs)

    # How many mini-batches (same ceil formula as original).
    n_batches = int(np.ceil(float(n) / float(batch_size))) if n > 0 else 0

    # Batch ids 0 .. n_batches-1
    batch_ids = list(range(n_batches))

    # Original: random.sample(range(n_batches), n_batches)
    if shuffle:
        batch_ids = random.sample(batch_ids, n_batches)

    # Yield one batch at a time.
    for index in batch_ids:
        start = index * batch_size
        end = min(start + batch_size, n)
        batch_seqs = seqs[start:end]
        batch_labels = labels[start:end]
        batch_times = None if times is None else times[start:end]
        yield batch_seqs, batch_labels, batch_times
