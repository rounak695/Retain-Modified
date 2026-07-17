"""
train.py — Train RETAIN with PyTorch

This is the modern replacement for ../retain.py.

What it does (high level):
  1. Read command-line options (data paths, sizes, learning settings)
  2. Build the RETAIN model
  3. Load train / validation / test data
  4. For many epochs: update weights on training batches, measure validation AUC
  5. Save the best model as a .pt file

Run from this folder (or with python path set), for example:

  python train.py my_visits 1000 my_labels ./outputs/model \\
      --simple_load --n_epochs 10 --batch_size 32
"""

# argparse = standard library for reading command-line flags.
import argparse

# os = helpers for file paths.
import os

# random = used only if we want extra shuffling control (NumPy also shuffles).
import random

# NumPy for numeric helpers (means, etc.).
import numpy as np

# PyTorch core.
import torch

# AUC score from scikit-learn (same metric as the original script).
from sklearn.metrics import roc_auc_score

# Our model and L2 helper from model.py in this same folder.
from model import RETAIN, retain_regularization_loss

# Data loading / padding helpers from data_utils.py in this same folder.
from data_utils import (
    load_data,
    load_data_simple,
    pad_batch,
    batch_to_tensors,
    iter_minibatches,
    load_pickle,
)


def append_log(message, log_path):
    """
    Print a message and also append it to a log file.

    Same idea as print2file() in the original retain.py.
    """
    # Show the message in the terminal.
    print(message)

    # Make sure the folder for the log file exists.
    folder = os.path.dirname(log_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # Append one line to the log file ('a' = append mode).
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def evaluate_auc(model, dataset, input_dim_size, batch_size, use_time, use_log_time, device):
    """
    Run the model on a dataset and compute ROC-AUC.

    AUC ≈ "how well do predicted scores rank positive patients above negative ones?"
    0.5 = random guessing, 1.0 = perfect ranking.
    """
    # Put the model in evaluation mode (turns OFF dropout).
    model.eval()

    # Collect all predicted scores and true labels.
    all_scores = []
    all_labels = []

    # torch.no_grad() = do not build a computation graph (faster, less memory).
    with torch.no_grad():
        # Walk through the dataset in mini-batches (no shuffle for evaluation).
        for batch_seqs, batch_labels, batch_times in iter_minibatches(dataset, batch_size, shuffle=False):
            # Pad this batch into arrays.
            x, t, lengths = pad_batch(
                batch_seqs,
                input_dim_size,
                times=batch_times if use_time else None,
                use_log_time=use_log_time,
            )

            # Convert to tensors on CPU or GPU.
            x_t, lengths_t, _, t_t = batch_to_tensors(x, lengths, t=t, device=device)

            # Forward pass → probabilities for each patient in the batch.
            scores = model(x_t, lengths_t, times=t_t)

            # Move scores back to CPU NumPy and store them.
            all_scores.extend(scores.detach().cpu().numpy().tolist())
            all_labels.extend(list(batch_labels))

    # If a split has only one class, roc_auc_score would crash; guard against that.
    if len(set(all_labels)) < 2:
        return 0.5

    # Compute and return AUC.
    return roc_auc_score(all_labels, all_scores)


def train_one_epoch(
    model,
    optimizer,
    dataset,
    input_dim_size,
    batch_size,
    use_time,
    use_log_time,
    device,
    l2_output,
    l2_alpha,
    l2_beta,
    l2_emb,
    log_eps,
    verbose,
):
    """
    Go through the training set once (one epoch) and update model weights.
    """
    # Enable dropout and training-time behavior.
    model.train()

    # Keep a list of loss values so we can average them for logging.
    cost_vector = []

    # How many mini-batches we will see (for verbose printing).
    n_batches = int(np.ceil(len(dataset[0]) / float(batch_size)))
    iteration = 0

    # Shuffle mini-batch order each epoch (like original random.sample).
    for batch_seqs, batch_labels, batch_times in iter_minibatches(dataset, batch_size, shuffle=True):
        # Build padded multi-hot matrices.
        x, t, lengths = pad_batch(
            batch_seqs,
            input_dim_size,
            times=batch_times if use_time else None,
            use_log_time=use_log_time,
        )

        # Convert labels to a plain float list/array for the tensor helper.
        y = np.asarray(batch_labels, dtype=np.float32)

        # Tensors on the chosen device.
        x_t, lengths_t, y_t, t_t = batch_to_tensors(x, lengths, y=y, t=t, device=device)

        # Clear old gradients from the previous step (required every iteration).
        optimizer.zero_grad()

        # Forward pass: predicted probabilities ŷ.
        y_hat = model(x_t, lengths_t, times=t_t)

        # Binary cross-entropy, matching the original formula with logEps.
        # We clamp ŷ away from 0 and 1 so log never sees exactly 0.
        y_hat_safe = y_hat.clamp(min=log_eps, max=1.0 - log_eps)
        bce = -(y_t * torch.log(y_hat_safe) + (1.0 - y_t) * torch.log(1.0 - y_hat_safe))
        # Mean over patients in the batch.
        loss = bce.mean()

        # Add L2 regularization on the same weights as the original script.
        loss = loss + retain_regularization_loss(model, l2_output, l2_alpha, l2_beta, l2_emb)

        # Backpropagation: compute d(loss)/d(each weight).
        loss.backward()

        # Apply the optimizer update (AdaDelta or Adam).
        optimizer.step()

        # Remember this batch's loss (as a Python float).
        cost_vector.append(float(loss.detach().cpu().item()))

        # Optional progress print every 10 mini-batches.
        if verbose and (iteration % 10 == 0):
            print(f"Iteration:{iteration}/{n_batches}, Train_Cost:{cost_vector[-1]:.6f}")

        iteration += 1

    # Return the average training loss for this epoch.
    return float(np.mean(cost_vector)) if cost_vector else 0.0


def train_retain(args):
    """
    Main training driver — wires data, model, optimizer, and the epoch loop.
    """
    # Choose GPU if available and requested; otherwise CPU.
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Whether this run uses time features.
    use_time = len(args.time_file) > 0

    # Create output directory if needed.
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Log file path: same prefix as the model output + ".log"
    log_file = args.out_file + ".log"

    # Optional pretrained embedding matrix from a pickle file.
    # NOTE: weight init uses NumPy's global RNG (same as original get_random_weight).
    # Seed was set in __main__ BEFORE this function; model build must happen
    # BEFORE load_data_simple() which reseeds np.random to 0 (same order as original).
    pretrained = None
    if len(args.embed_file) > 0:
        print("using external code embedding")
        pretrained = np.array(load_pickle(args.embed_file), dtype=np.float32)
        # If pretrained is provided, its second dimension is the embedding size.
        emb_dim_size = pretrained.shape[1]
    else:
        print("using randomly initialized code embedding")
        emb_dim_size = args.embed_size

    # Build the RETAIN model (NumPy draws weights here, in init_params order).
    model = RETAIN(
        input_dim_size=args.n_input_codes,
        emb_dim_size=emb_dim_size,
        alpha_hidden_size=args.alpha_hidden_dim_size,
        beta_hidden_size=args.beta_hidden_dim_size,
        use_time=use_time,
        emb_finetune=bool(args.embed_finetune),
        keep_prob_emb=args.keep_prob_emb,
        keep_prob_context=args.keep_prob_context,
        pretrained_embedding=pretrained,
    ).to(device)

    # Optionally resume from a previous PyTorch checkpoint (.pt).
    if len(args.model_file) > 0:
        print(f"Loading checkpoint from {args.model_file}")
        state = torch.load(args.model_file, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    # Optimizers matched to the original Theano retain.py.
    if args.solver == "adadelta":
        # Original adadelta(): rho=0.95, eps=1e-6, update scale = 1 (no extra lr).
        # PyTorch lr=1.0 reproduces that unscaled update.
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=1.0, rho=0.95, eps=1e-6, weight_decay=0.0
        )
    elif args.solver == "adam":
        # Original adam(b1=0.1, b2=0.001) is the SAME math as standard Adam
        # with betas=(0.9, 0.999), because those b1/b2 were (1 - beta).
        # lr=0.0002 matches the original default.
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

    # Load datasets AFTER model init (original calls init_params before load_data;
    # load_data_simple reseeds np.random to 0 for the split).
    print("Loading data ...")
    if args.simple_load:
        train_set, valid_set, test_set = load_data_simple(args.seq_file, args.label_file, args.time_file)
    else:
        train_set, valid_set, test_set = load_data(args.seq_file, args.label_file, args.time_file)
    print("done")

    # Track the best validation AUC seen so far.
    best_valid_auc = 0.0
    best_test_auc = 0.0
    best_valid_epoch = 0

    append_log("Optimization start !!", log_file)

    # Main training loop over epochs.
    for epoch in range(args.n_epochs):
        # Train on all training mini-batches once.
        train_cost = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataset=train_set,
            input_dim_size=args.n_input_codes,
            batch_size=args.batch_size,
            use_time=use_time,
            use_log_time=bool(args.use_log_time),
            device=device,
            l2_output=args.L2_output,
            l2_alpha=args.L2_alpha,
            l2_beta=args.L2_beta,
            l2_emb=args.L2_emb,
            log_eps=args.log_eps,
            verbose=args.verbose,
        )

        # Measure validation AUC after the epoch.
        valid_auc = evaluate_auc(
            model,
            valid_set,
            args.n_input_codes,
            args.batch_size,
            use_time,
            bool(args.use_log_time),
            device,
        )

        # Log train cost + validation AUC.
        buf = f"Epoch:{epoch}, Train_cost:{train_cost:.6f}, Validation_AUC:{valid_auc:.6f}"
        append_log(buf, log_file)

        # If this is the best validation AUC so far, save a checkpoint.
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_valid_epoch = epoch

            # Also evaluate on the test set (same protocol as the original script).
            best_test_auc = evaluate_auc(
                model,
                test_set,
                args.n_input_codes,
                args.batch_size,
                use_time,
                bool(args.use_log_time),
                device,
            )

            buf = f"Currently the best validation AUC found. Test AUC:{best_test_auc:.6f} at epoch:{epoch}"
            append_log(buf, log_file)

            # Save model weights + a few settings needed later for interpretation.
            ckpt_path = f"{args.out_file}.{epoch}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim_size": args.n_input_codes,
                    "emb_dim_size": emb_dim_size,
                    "alpha_hidden_size": args.alpha_hidden_dim_size,
                    "beta_hidden_size": args.beta_hidden_dim_size,
                    "use_time": use_time,
                    "use_log_time": bool(args.use_log_time),
                    "emb_finetune": bool(args.embed_finetune),
                    "keep_prob_emb": args.keep_prob_emb,
                    "keep_prob_context": args.keep_prob_context,
                    "epoch": epoch,
                    "valid_auc": valid_auc,
                    "test_auc": best_test_auc,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    # Final summary line.
    buf = (
        f"The best validation & test AUC:{best_valid_auc:.6f}, "
        f"{best_test_auc:.6f} at epoch:{best_valid_epoch}"
    )
    append_log(buf, log_file)


def parse_arguments():
    """
    Define and parse command-line arguments (mirrors original retain.py flags).
    """
    parser = argparse.ArgumentParser(description="Train RETAIN (PyTorch rewrite)")

    # Positional arguments (required).
    parser.add_argument("seq_file", type=str, help="Visit file prefix (or full path with --simple_load)")
    parser.add_argument("n_input_codes", type=int, help="Number of unique medical codes")
    parser.add_argument("label_file", type=str, help="Label file prefix (or full path with --simple_load)")
    parser.add_argument("out_file", type=str, help="Output path prefix for checkpoints and logs")

    # Optional flags (same names as the original where possible).
    parser.add_argument("--time_file", type=str, default="", help="Optional time file prefix/path")
    parser.add_argument("--model_file", type=str, default="", help="Optional .pt checkpoint to resume")
    parser.add_argument("--use_log_time", type=int, default=1, choices=[0, 1], help="log(time+1) if 1")
    parser.add_argument("--embed_file", type=str, default="", help="Optional pretrained embedding pickle")
    parser.add_argument("--embed_size", type=int, default=128, help="Embedding size if not using embed_file")
    parser.add_argument("--embed_finetune", type=int, default=1, choices=[0, 1], help="Fine-tune embeddings")
    parser.add_argument("--alpha_hidden_dim_size", type=int, default=128, help="GRU_alpha hidden size")
    parser.add_argument("--beta_hidden_dim_size", type=int, default=128, help="GRU_beta hidden size")
    parser.add_argument("--batch_size", type=int, default=100, help="Mini-batch size")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--L2_output", type=float, default=0.001, help="L2 on classifier weight")
    parser.add_argument("--L2_emb", type=float, default=0.001, help="L2 on embedding weight")
    parser.add_argument("--L2_alpha", type=float, default=0.001, help="L2 on w_alpha")
    parser.add_argument("--L2_beta", type=float, default=0.001, help="L2 on W_beta")
    parser.add_argument("--keep_prob_emb", type=float, default=0.5, help="Keep prob for embedding dropout")
    parser.add_argument("--keep_prob_context", type=float, default=0.5, help="Keep prob for context dropout")
    parser.add_argument("--log_eps", type=float, default=1e-8, help="Epsilon to avoid log(0)")
    parser.add_argument("--solver", type=str, default="adadelta", choices=["adadelta", "adam"])
    parser.add_argument("--simple_load", action="store_true", help="Auto-split a single dataset file")
    parser.add_argument("--verbose", action="store_true", help="Print every 10 mini-batches")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    # This block runs only when you execute: python train.py ...
    args = parse_arguments()

    # Seed RNGs the same way the original stack expects:
    #   - np.random  → weight init (get_random_weight) AND simple_load split
    #   - random     → mini-batch order (random.sample)
    # Dropout uses a separate NumPy RandomState(1234) inside the model
    # (mirrors Theano RandomStreams(1234)).
    # torch seed is only a fallback; weights/masks do not use torch RNG.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Start training.
    train_retain(args)
