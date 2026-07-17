"""
model.py — RETAIN neural network in PyTorch

Faithful port of ../retain.py (gru_layer + build_model) and ../test_retain.py.

Matches the original on:
  - weight init via NumPy uniform(-0.1, 0.1) in the SAME order as init_params()
  - dropout masks via NumPy binomial (stream seeded 1234 like Theano RandomStreams(1234))
  - custom GRU gate equations
  - reverse-time attention recomputed for EVERY prefix length (attentionStep scan)
  - prediction taken at lengths - 1
  - contribution formula

Read ALGORITHM.md for the big-picture math.
"""

# NumPy is used for RNG so weight/mask draws match the original (not PyTorch RNG).
import numpy as np

# Import PyTorch's main package and give it the short name "torch".
import torch

# Import the neural-network building blocks (layers, Module base class, etc.).
import torch.nn as nn


def get_random_weight(dim1, dim2, left=-0.1, right=0.1):
    """
    Exact clone of get_random_weight() in the original retain.py.

    Uses the global NumPy RNG (np.random), so seeding with np.random.seed(...)
    before building the model controls these draws the same way as Theano RETAIN.
    """
    return np.random.uniform(left, right, (dim1, dim2)).astype(np.float32)


def _set_linear_weight(linear, W_in_out, bias_vec=None):
    """
    Copy a NumPy matrix into an nn.Linear.

    Original stores W with shape (in_features, out_features).
    nn.Linear stores weight with shape (out_features, in_features) = W.T.
    """
    with torch.no_grad():
        # Contiguous float32 tensor, transposed for Linear layout.
        linear.weight.copy_(torch.from_numpy(np.ascontiguousarray(W_in_out.T)))
        if bias_vec is not None and linear.bias is not None:
            linear.bias.copy_(torch.from_numpy(np.ascontiguousarray(bias_vec.astype(np.float32))))


def retain_dropout(x, keep_prob, training, rng):
    """
    Dropout matching theano dropout_layer() + MRG_RandomStreams draws:

      training:  x * Bernoulli(keep_prob) / keep_prob
      eval:      x unchanged

    Masks are drawn with NumPy RandomState.binomial (not torch.bernoulli),
    so the RNG family matches the original NumPy/Theano side.
    """
    # If keep everything, or we are not training, do nothing.
    if (not training) or keep_prob >= 1.0:
        return x
    # If keep nothing, return zeros (edge case).
    if keep_prob <= 0.0:
        return x * 0.0

    # NumPy binomial(n=1, p=keep_prob) == Bernoulli(keep_prob), same as Theano.
    mask_np = rng.binomial(n=1, p=keep_prob, size=tuple(x.shape)).astype(np.float32)
    # Move mask to the same device/dtype as x (CPU or GPU).
    mask = torch.from_numpy(mask_np).to(device=x.device, dtype=x.dtype)
    # Scale like the original so expected value stays the same.
    return x * mask / keep_prob


class CustomGRU(nn.Module):
    """
    Custom GRU matching Edward Choi's Theano gru_layer exactly.

    Gates (same order as original _slice 0,1,2):
      r = σ(W_r x + U_r h)
      z = σ(W_z x + U_z h)
      h̃ = tanh(W_h x + r ⊙ U_h h)
      h' = z ⊙ h + (1 − z) ⊙ h̃

    Weights are filled later by RETAIN.init_params_like_original() using NumPy.
    """

    def __init__(self, input_size, hidden_size):
        # Always call the parent constructor first when subclassing nn.Module.
        super().__init__()

        # Remember how big the hidden state is; we need it when slicing gates.
        self.hidden_size = hidden_size

        # W_gru: (input_size → 3*hidden) with bias  — original W_gru_* + b_gru_*
        self.W_gru = nn.Linear(input_size, 3 * hidden_size, bias=True)

        # U_gru: (hidden → 3*hidden) WITHOUT bias — original U_gru_* has no bias
        self.U_gru = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x):
        """
        x shape: (batch, time, input_size)  — already in the order to read
        returns: (batch, time, hidden_size)
        """
        # Read batch size and number of time steps from the input tensor.
        batch_size, n_timesteps, _ = x.shape

        # Start with a memory of all zeros (Theano: T.alloc(0, n_samples, hidden)).
        h = x.new_zeros(batch_size, self.hidden_size)

        # Collect hidden state after every visit.
        outputs = []

        # Loop over the given order (caller reverses time for RETAIN).
        for t in range(n_timesteps):
            # Visit embedding at time t for every patient: (batch, input_size)
            x_t = x[:, t, :]

            # Wx = x @ W + b   (original: T.dot(emb, W_gru) + b_gru)
            wx = self.W_gru(x_t)

            # Uh = h @ U
            uh = self.U_gru(h)

            # Split into reset / update / candidate blocks (same as _slice).
            wx_r, wx_z, wx_h = wx.split(self.hidden_size, dim=-1)
            uh_r, uh_z, uh_h = uh.split(self.hidden_size, dim=-1)

            # Reset gate.
            r = torch.sigmoid(wx_r + uh_r)
            # Update gate.
            z = torch.sigmoid(wx_z + uh_z)
            # Candidate memory.
            h_tilde = torch.tanh(wx_h + r * uh_h)
            # Mix old and new (original: z*h + (1-z)*h_tilde).
            h = z * h + (1.0 - z) * h_tilde

            outputs.append(h)

        # Stack → (batch, time, hidden)
        return torch.stack(outputs, dim=1)


class RETAIN(nn.Module):
    """
    Full RETAIN model mirroring build_model() in the original retain.py.
    """

    def __init__(
        self,
        input_dim_size,
        emb_dim_size=128,
        alpha_hidden_size=128,
        beta_hidden_size=128,
        use_time=False,
        emb_finetune=True,
        keep_prob_emb=0.5,
        keep_prob_context=0.5,
        pretrained_embedding=None,
    ):
        # Initialize nn.Module bookkeeping.
        super().__init__()

        # Store settings for forward().
        self.input_dim_size = input_dim_size
        self.emb_dim_size = emb_dim_size
        self.alpha_hidden_size = alpha_hidden_size
        self.beta_hidden_size = beta_hidden_size
        self.use_time = use_time
        self.emb_finetune = emb_finetune
        # Keep original naming: probability of KEEPING a unit.
        self.keep_prob_emb = keep_prob_emb
        self.keep_prob_context = keep_prob_context

        # Dropout RNG: mirrors `trng = RandomStreams(1234)` in build_model().
        # Fixed seed 1234 so dropout draws are NumPy-based and reproducible.
        self._dropout_rng = np.random.RandomState(1234)

        # v = x @ W_emb   (no bias) — Linear weight is (emb, codes), so forward does x @ W.T
        self.embedding = nn.Linear(input_dim_size, emb_dim_size, bias=False)

        # GRU input size = emb, or emb+1 with time.
        gru_input_size = emb_dim_size + (1 if use_time else 0)
        self._gru_input_size = gru_input_size

        # Two GRUs: alpha (visit) and beta (variable).
        self.gru_alpha = CustomGRU(gru_input_size, alpha_hidden_size)
        self.gru_beta = CustomGRU(gru_input_size, beta_hidden_size)

        # w_alpha, b_alpha
        self.w_alpha = nn.Linear(alpha_hidden_size, 1, bias=True)

        # W_beta, b_beta
        self.W_beta = nn.Linear(beta_hidden_size, emb_dim_size, bias=True)

        # w_output, b_output
        self.w_output = nn.Linear(emb_dim_size, 1, bias=True)

        # Fill every weight in the SAME ORDER as original init_params().
        self._init_params_like_original(pretrained_embedding)

        # Freeze embedding when not fine-tuning (same as skipping W_emb in tparams).
        if not emb_finetune:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def _init_params_like_original(self, pretrained_embedding):
        """
        Mirror init_params() draw order exactly:

          W_emb (or load), W_gru_a, U_gru_a, b_gru_a,
          W_gru_b, U_gru_b, b_gru_b,
          w_alpha, b_alpha, W_beta, b_beta, w_output, b_output

        All random matrices come from NumPy get_random_weight / np.random.uniform.
        """
        # --- W_emb ---
        if pretrained_embedding is not None:
            W_emb = np.array(pretrained_embedding, dtype=np.float32)
            # External file sets emb size; caller already passed matching emb_dim_size.
        else:
            # get_random_weight(inputDimSize, embDimSize)
            W_emb = get_random_weight(self.input_dim_size, self.emb_dim_size)
        _set_linear_weight(self.embedding, W_emb)

        # --- GRU alpha: W_gru_a, U_gru_a, b_gru_a ---
        W_gru_a = get_random_weight(self._gru_input_size, 3 * self.alpha_hidden_size)
        U_gru_a = get_random_weight(self.alpha_hidden_size, 3 * self.alpha_hidden_size)
        b_gru_a = np.zeros(3 * self.alpha_hidden_size, dtype=np.float32)
        _set_linear_weight(self.gru_alpha.W_gru, W_gru_a, b_gru_a)
        _set_linear_weight(self.gru_alpha.U_gru, U_gru_a)

        # --- GRU beta: W_gru_b, U_gru_b, b_gru_b ---
        W_gru_b = get_random_weight(self._gru_input_size, 3 * self.beta_hidden_size)
        U_gru_b = get_random_weight(self.beta_hidden_size, 3 * self.beta_hidden_size)
        b_gru_b = np.zeros(3 * self.beta_hidden_size, dtype=np.float32)
        _set_linear_weight(self.gru_beta.W_gru, W_gru_b, b_gru_b)
        _set_linear_weight(self.gru_beta.U_gru, U_gru_b)

        # --- attention + output ---
        w_alpha = get_random_weight(self.alpha_hidden_size, 1)
        b_alpha = np.zeros(1, dtype=np.float32)
        _set_linear_weight(self.w_alpha, w_alpha, b_alpha)

        W_beta = get_random_weight(self.beta_hidden_size, self.emb_dim_size)
        b_beta = np.zeros(self.emb_dim_size, dtype=np.float32)
        _set_linear_weight(self.W_beta, W_beta, b_beta)

        w_output = get_random_weight(self.emb_dim_size, 1)
        b_output = np.zeros(1, dtype=np.float32)
        _set_linear_weight(self.w_output, w_output, b_output)

    def embed_visits(self, x):
        """
        Multi-hot visits → dense embeddings (+ optional emb dropout).

        x: (batch, time, num_codes)
        """
        emb = self.embedding(x)
        # Original: only when keep_prob_emb < 1.0 and use_noise=1 (training).
        if self.keep_prob_emb < 1.0:
            emb = retain_dropout(emb, self.keep_prob_emb, self.training, self._dropout_rng)
        return emb

    def attention_step(self, temb_prefix, emb_prefix):
        """
        One attentionStep(att_timesteps) from the original Theano graph.

        Takes ONLY the first att_timesteps visits (already sliced by caller):
          1) reverse time
          2) run both GRUs
          3) reverse hidden states back
          4) × 0.5
          5) alpha softmax, beta tanh
          6) c = sum_i alpha_i * (beta_i ⊙ emb_i)

        temb_prefix: (batch, t, gru_input)  chronological
        emb_prefix:  (batch, t, emb_dim)    chronological (no time feature)
        returns:     c  (batch, emb_dim)
                     also alpha (batch, t), beta (batch, t, emb) for interpret
        """
        # reverse_emb_t = temb[:att][::-1]
        rev = temb_prefix.flip(dims=[1])

        # GRU on reversed visits, then flip outputs back to chronological order.
        h_a = self.gru_alpha(rev).flip(dims=[1]) * 0.5
        h_b = self.gru_beta(rev).flip(dims=[1]) * 0.5

        # preAlpha → softmax over visits (original: softmax(preAlpha.T).T)
        pre_alpha = self.w_alpha(h_a).squeeze(-1)  # (batch, t)
        alpha = torch.softmax(pre_alpha, dim=1)

        # beta = tanh(h_b @ W_beta + b_beta)
        beta = torch.tanh(self.W_beta(h_b))

        # c_t = sum over visits of alpha * beta * emb
        c = (alpha.unsqueeze(-1) * beta * emb_prefix).sum(dim=1)
        return c, alpha, beta

    def forward(self, x, lengths, times=None, return_attention=False):
        """
        Full forward pass matching build_model().

        x:       (batch, time, num_codes)
        lengths: (batch,)
        times:   (batch, time) optional

        Training/eval prediction path:
          - run attentionStep for prefix lengths 1..T (Theano scan)
          - apply context dropout to the stack of context vectors
          - sigmoid classifier at every prefix
          - pick y_hat at index (lengths - 1) for each patient

        return_attention=True (interpretation, like test_retain.py):
          - run one attention pass on each patient's full real history
        """
        # Embed visits.
        emb = self.embed_visits(x)

        # Optional time concat → temb.
        if self.use_time:
            if times is None:
                raise ValueError("use_time=True but times was not provided")
            temb = torch.cat([emb, times.unsqueeze(-1)], dim=-1)
        else:
            temb = emb

        batch_size, n_timesteps, _ = emb.shape

        # ---- Interpretation path (test_retain): one full reverse pass ----
        if return_attention:
            # For each patient, use exactly their real visit count (no pads),
            # same as feeding one unpadded patient through test_retain.
            alphas = []
            betas = []
            y_list = []
            for i in range(batch_size):
                L = int(lengths[i].item())
                # Safety: empty history → probability 0.5-ish via zero context.
                if L <= 0:
                    c = emb.new_zeros(self.emb_dim_size)
                    alpha_i = emb.new_zeros(n_timesteps)
                    beta_i = emb.new_zeros(n_timesteps, self.emb_dim_size)
                else:
                    c, alpha_pref, beta_pref = self.attention_step(
                        temb[i : i + 1, :L, :],
                        emb[i : i + 1, :L, :],
                    )
                    c = c[0]
                    # Pad alpha/beta back to full T for a consistent return shape.
                    alpha_i = emb.new_zeros(n_timesteps)
                    beta_i = emb.new_zeros(n_timesteps, self.emb_dim_size)
                    alpha_i[:L] = alpha_pref[0]
                    beta_i[:L] = beta_pref[0]

                # No context dropout at interpret time (use_noise=0 / eval).
                y_i = torch.sigmoid(self.w_output(c).squeeze(-1))
                y_list.append(y_i)
                alphas.append(alpha_i)
                betas.append(beta_i)

            y_hat = torch.stack(y_list, dim=0)
            alpha = torch.stack(alphas, dim=0)
            beta = torch.stack(betas, dim=0)
            # mask of real visits
            time_idx = torch.arange(n_timesteps, device=emb.device).unsqueeze(0)
            mask = time_idx < lengths.unsqueeze(1)
            return y_hat, alpha, beta, emb, mask

        # ---- Training path: Theano scan over prefix lengths 1..T ----
        # c_stack will be (batch, T, emb_dim) — one context per prefix length.
        c_list = []
        for att_timesteps in range(1, n_timesteps + 1):
            # attentionStep(att_timesteps): use first att_timesteps visits only.
            c_t, _, _ = self.attention_step(
                temb[:, :att_timesteps, :],
                emb[:, :att_timesteps, :],
            )
            c_list.append(c_t)

        # Stack along time: (batch, T, emb_dim)
        c_stack = torch.stack(c_list, dim=1)

        # Context dropout (original applies dropout on c_t before classifier).
        if self.keep_prob_context < 1.0:
            c_stack = retain_dropout(
                c_stack, self.keep_prob_context, self.training, self._dropout_rng
            )

        # preY for every prefix: (batch, T)
        pre_y = torch.sigmoid(self.w_output(c_stack).squeeze(-1))

        # y_hat = preY[patient, lengths - 1]  — exact original indexing
        index_row = torch.arange(batch_size, device=emb.device)
        y_hat = pre_y[index_row, lengths - 1]

        return y_hat

    def contribution_for_code(self, alpha_i, beta_i, code_index):
        """
        contrib = w_output · (alpha_i * beta_i * W_emb[code])
        Same formula as test_retain.py.
        """
        # Embedding of this code: column of Linear weight (emb_dim,).
        code_emb = self.embedding.weight[:, code_index]
        attended = alpha_i * beta_i * code_emb
        w = self.w_output.weight.squeeze(0)
        return torch.dot(w, attended)


def retain_regularization_loss(model, l2_output, l2_alpha, l2_beta, l2_emb):
    """
    Same L2 terms as original cost:
      L2_output * ||w_output||^2 + L2_alpha * ||w_alpha||^2
      + L2_beta * ||W_beta||^2 + (optional) L2_emb * ||W_emb||^2
    """
    reg = model.w_output.weight.pow(2).sum() * l2_output
    reg = reg + model.w_alpha.weight.pow(2).sum() * l2_alpha
    reg = reg + model.W_beta.weight.pow(2).sum() * l2_beta
    if model.emb_finetune and l2_emb > 0:
        reg = reg + model.embedding.weight.pow(2).sum() * l2_emb
    return reg
