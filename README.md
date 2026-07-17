# Retain-Modified

A faithful **Python 3 / PyTorch** reimplementation of **RETAIN** (*REverse Time AttentIoN*), the interpretable recurrent model introduced by Choi et al. for prediction from electronic health records.

**Reference.** Edward Choi, Mohammad Taha Bahadori, Joshua A. Kulas, Andy Schuetz, Walter F. Stewart, Jimeng Sun. *RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism.* NeurIPS 2016.  
https://papers.nips.cc/paper/6321-retain-an-interpretable-predictive-model-for-healthcare-using-reverse-time-attention-mechanism

**Upstream (Theano / Python 2).** https://github.com/mp2893/retain

This repository keeps the **scientific model** of the original code—equations, training objective, attention schedule, and interpretation rule—while replacing a discontinued deep-learning stack with a maintainable one.

---

## What RETAIN does

Given a patient’s sequence of visits (each visit a set of medical codes), RETAIN:

1. Embeds codes into dense vectors  
2. Reads the history **backwards in time** with two GRUs  
3. Forms **visit-level** attention $\alpha$ and **variable-level** attention $\beta$  
4. Builds a context vector and outputs a risk probability  
5. Explains the score by attributing contribution to each code at each visit  

This codebase implements the **sequence-classification** special case used in the original scripts: one label per patient (for example mortality or future diagnosis), predicted from the full history.

---

## Repository layout

| File | Role |
|------|------|
| `model.py` | RETAIN network (custom GRU, prefix attention, prediction, contributions) |
| `data_utils.py` | Pickle I/O, multi-hot padding, batching |
| `train.py` | Training loop, validation AUC, checkpointing |
| `interpret.py` | Per-code contribution reports |
| `process_mimic.py` | MIMIC-III → RETAIN pickle pipeline (Python 3) |
| `requirements.txt` | Runtime dependencies |

Source files include line-level comments aimed at readers with basic Python.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.9+**, **PyTorch ≥ 2**, **NumPy**, **scikit-learn**.

---

## Data format

Identical to the original RETAIN contract.

| Artifact | Structure |
|----------|-----------|
| Visits | `list[list[list[int]]]` — patient → visits → integer code ids |
| Labels | `list[int]` — binary outcome per patient |
| Types (interpretation) | `dict[str, int]` — code string → id |
| Times (optional) | `list[list[float]]` — time feature per visit |

Default loader expects suffixes `.train` / `.valid` / `.test`.  
`--simple_load` splits a single visit/label pair (20% test, 10% validation, remainder train; `np.random.seed(0)` inside the split, as in the original).

---

## Training

```bash
python train.py <visit_file> <#unique_codes> <label_file> <output_prefix> [options]
```

Example:

```bash
python train.py path/to/data.3digitICD9.seqs 942 path/to/data.morts ./outputs/model \
  --simple_load --n_epochs 100 --keep_prob_context 0.8 --keep_prob_emb 0.5 --seed 0
```

Useful flags: `--time_file`, `--use_log_time`, `--embed_size`, `--alpha_hidden_dim_size`, `--beta_hidden_dim_size`, `--batch_size`, `--solver {adadelta,adam}`, `--cuda`, `--verbose`, `--seed`.

Checkpoints are written as `<output_prefix>.<epoch>.pt` when validation AUC improves.

---

## Interpretation

```bash
python interpret.py <model.pt> <visit_file> <label_file> <types_file> <output.txt>
```

Each line reports a code’s signed contribution to the predicted score at a given visit.

---

## MIMIC-III preprocessing

```bash
python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv <output_prefix>
```

---

## Algorithm (concise)

### Embedding

A multi-hot visit vector $x_i \in \{0,1\}^N$ is mapped to a dense embedding:

$$v_i = x_i W_{\mathrm{emb}}$$

An optional time feature may be concatenated **before** the GRUs. Only the embeddings $v_i$ (without the time channel) enter the context sum below.

### Reverse-time GRUs

For each prefix length $t = 1, \ldots, T$:

1. Take the first $t$ visits and reverse them in time  
2. Run $\mathrm{GRU}_{\alpha}$ and $\mathrm{GRU}_{\beta}$  
3. Reverse the hidden states back to chronological order  
4. Scale hidden states by $\frac{1}{2}$ (as in the reference implementation)

### Attention

Visit-level weights $\alpha$ and variable-level weights $\beta$:

$$\alpha = \mathrm{softmax}\big(w_{\alpha}^{\top} h^{(\alpha)} + b_{\alpha}\big)$$

$$\beta = \tanh\big(h^{(\beta)} W_{\beta} + b_{\beta}\big)$$

### Context and prediction

$$c_t = \sum_{i=1}^{t} \alpha_i\,(\beta_i \odot v_i)$$

$$\hat{y} = \sigma\big(w^{\top} c_L + b\big)$$

Here $L$ is the patient’s true history length (indexing `lengths - 1` after the prefix scan), $\odot$ is element-wise multiplication, and $\sigma$ is the sigmoid.

### Objective

Mean binary cross-entropy with an $\varepsilon$-stable log, plus L2 penalties on $w_{\mathrm{output}}$, $w_{\alpha}$, $W_{\beta}$, and optionally $W_{\mathrm{emb}}$:

$$\mathcal{L} = -\frac{1}{B}\sum_{b=1}^{B}\Big[y_b\log(\hat{y}_b+\varepsilon)+(1-y_b)\log(1-\hat{y}_b+\varepsilon)\Big] + \lambda_o\lVert w_{\mathrm{output}}\rVert_2^2 + \lambda_{\alpha}\lVert w_{\alpha}\rVert_2^2 + \lambda_{\beta}\lVert W_{\beta}\rVert_2^2 + \lambda_e\lVert W_{\mathrm{emb}}\rVert_2^2$$

(The last term is used only when embeddings are fine-tuned.)

### Contribution of code $j$ at visit $i$

$$\mathrm{contrib}(i,j) = w^{\top}\big(\alpha_i \cdot \beta_i \odot W_{\mathrm{emb}}[j]\big)$$

Positive values push $\hat{y}$ up; negative values push it down.

### Code map

| Step | Code |
|------|------|
| Embed | `RETAIN.embed_visits` |
| GRU | `CustomGRU` |
| Prefix attention | `attention_step` / `forward` |
| Contributions | `contribution_for_code` / `interpret.py` |

---

## Equivalence to the Theano original

### Matched on purpose

| Component | Behavior |
|-----------|----------|
| Task | Sequence classification (one $\hat{y}$ per patient) |
| Input / pickle schema | Same visit, label, time, and types layouts |
| Multi-hot padding + `log(time+1)` | Same |
| Weight init | `np.random.uniform(-0.1, 0.1)` in `init_params` order; biases zero |
| Custom GRU gates | Identical reset / update / candidate equations |
| Prefix `attentionStep` scan | Recompute attention for $t=1,\ldots,T$; select `lengths-1` |
| $\alpha$ / $\beta$ / context / sigmoid | Same formulas |
| Dropout | Keep-probability Bernoulli, scaled by $1/\mathrm{keep\_prob}$ |
| Dropout stream seed | `1234` (NumPy `RandomState`), analogous to `RandomStreams(1234)` |
| Batch order | `random.sample(range(n_batches), n_batches)` |
| Loss + L2 groups | Same |
| AdaDelta | $\rho=0.95$, $\varepsilon=10^{-6}$, effective step scale $1$ |
| Adam | Equivalent to original `b1=0.1`, `b2=0.001` parameterization ($\beta=(0.9,0.999)$), $\mathrm{lr}=2\times 10^{-4}$ |
| Interpretation formula | Same dot-product contribution |
| Train/init order | Build parameters before `simple_load` (which reseeds NumPy to `0` for the split) |

### Minute differences that remain

These are the **only** intentional or unavoidable deviations. None of them change the mathematical definition of RETAIN in this codebase; they affect engineering, numerics, or cross-framework reproducibility.

1. **Language and framework**  
   Original: Python 2.7 + Theano 0.8.  
   This repo: Python 3 + PyTorch. Theano is unmaintained; this is the reason for the port.

2. **Serialization**  
   Original checkpoints: `numpy.savez_compressed` (`.npz`).  
   This repo: `torch.save` state dict (`.pt`).  
   **Old `.npz` weights are not loaded automatically.** Retrain, or write a one-off converter if you must migrate a specific file.

3. **Pickle module**  
   Original: `cPickle`.  
   This repo: stdlib `pickle`.  
   Data *content* is the same; binary pickle protocol bytes may differ across Python major versions when rewriting files.

4. **Tensor layout in memory**  
   Original graphs are largely **time-major** `(T, batch, features)`.  
   This repo uses **batch-first** `(batch, T, features)`, which is standard in PyTorch.  
   Algebra is the same; only axis order differs.

5. **Dropout generator family**  
   Original: Theano `MRG_RandomStreams(1234)` (MRG31k3p).  
   This repo: `numpy.random.RandomState(1234).binomial(...)`.  
   Same *interface* (Bernoulli keep-prob, seed 1234) and same *intent*, but **MRG ≠ NumPy MT19937**, so a step-by-step mask sequence will not match Theano bit-for-bit. Weight init *does* use NumPy, matching the original `get_random_weight`.

6. **Floating-point kernels**  
   `sigmoid`, `tanh`, `softmax`, GEMM, and reductions are executed by PyTorch (and optionally cuDNN) rather than Theano’s compiled graph. Expect **ulps-level** differences even with identical weights.

7. **AdaDelta / Adam implementation surface**  
   Update *rules* are aligned (see table above). The original uses hand-written Theano shared-variable updates; this repo uses `torch.optim.Adadelta` / `Adam`. Accumulator ordering and fused kernels can introduce tiny state differences over long training.

8. **Device and dtype defaults**  
   Original often ran under Theano `floatX` (commonly `float32`) on CPU/GPU via Theano config.  
   This repo defaults to PyTorch `float32` on CPU, or CUDA when `--cuda` is set. GPU nondeterminism (atomic adds, algorithm selection) can add run-to-run variance unless you enforce deterministic CuDNN settings yourself.

9. **Graph construction style**  
   Original builds a symbolic Theano graph once (`theano.scan` over prefixes).  
   This repo runs an explicit Python loop over prefix lengths in `forward`.  
   Semantically equivalent to the scan; performance and autograd tape structure differ.

10. **Interpretation driver**  
    Original `test_retain.py` feeds already-embedded (+ time) tensors into a slim Theano graph.  
    This repo’s `interpret.py` calls `RETAIN(..., return_attention=True)`, which applies the same attention math on each patient’s real-length prefix. Outputs follow the same contribution formula.

11. **CLI extras**  
    This repo adds `--seed` and `--cuda`. Defaults for model hyperparameters match the original where names overlap.

12. **Packaging**  
    Only the modified PyTorch RETAIN lives here. The upstream Theano tree, demos, and figures are not vendored.

### Practical reproducibility guidance

- Fix `--seed` for NumPy weight draws and Python `random` batch order.  
- Dropout uses a separate NumPy stream seeded at `1234` inside the model.  
- Do not expect bit-identical loss curves versus a Theano run on another machine; do expect the **same model, objective, and interpretation semantics**.  
- For the strongest cross-check: fix weights, disable dropout (`keep_prob_*=1`), run both forwards on one batch, and compare $\hat{y}$ within floating-point tolerance.

---

## Citation

If you use RETAIN in research, cite the original paper:

```bibtex
@inproceedings{choi2016retain,
  title     = {RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism},
  author    = {Choi, Edward and Bahadori, Mohammad Taha and Kulas, Joshua A and Schuetz, Andy and Stewart, Walter F and Sun, Jimeng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2016}
}
```

Please also acknowledge the upstream implementation at https://github.com/mp2893/retain when comparing against or extending that codebase.

---

## License note

Follow the license terms of the upstream RETAIN project for derivative use of the algorithm and any code adapted from it. This repository is a framework modernization of that work.
