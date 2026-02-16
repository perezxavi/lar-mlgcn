import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from typing import TypedDict


class GraphStats(TypedDict):
    density: float
    mean_degree: float
    degrees: torch.Tensor
    E: int
    C: int


class LaRMLGCNClassifier(nn.Module):
    """ML-GCN classifier for tabular multi-label problems.

    Architecture:
    - Label path (GCN): label embeddings ``Z`` are propagated over a fixed
      label graph and projected into class-wise classifiers ``W``.
    - Feature path (MLP): input features ``X`` are projected into ``Phi``.
    - Logits: ``Phi @ W.T (+ class_bias)``.

    The label graph is built once from training labels via ``build_label_graph``
    and stored as buffer ``A_hat``.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        proj_dim: int = 128,
        mlp_hidden_dims: tuple[int, ...] = (256,),
        mlp_batchnorm: bool = True,
        mlp_dropout: float = 0.0,
        label_embed_dim: int = 32,
        gcn_hidden_dims: tuple[int, ...] = (128,),
        tau: float = 0.4,
        p: float = 0.2,
        use_class_bias: bool = True,
        negative_slope: float = 0.2,
        device: str | None = None,
        alpha: float = 1.0,
        gcn_do_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = self._validate_positive_int(input_dim, "input_dim")
        self.C = self._validate_positive_int(num_labels, "num_labels")
        self.F = self._validate_positive_int(proj_dim, "proj_dim")
        self.d = self._validate_positive_int(label_embed_dim, "label_embed_dim")

        self.tau = self._validate_unit_interval(tau, "tau")
        self.p = self._validate_unit_interval(p, "p")
        self.neg_slope = float(negative_slope)
        self.alpha = self._validate_non_negative(alpha, "alpha")
        self.gcn_do_p = self._validate_unit_interval(gcn_do_p, "gcn_do_p")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.Z = nn.Parameter(torch.randn(self.C, self.d) * 0.02)
        self.gcn_layers = self._build_gcn_layers(gcn_hidden_dims)
        self.act = nn.LeakyReLU(negative_slope=self.neg_slope)

        self.feature_mlp = self._build_feature_mlp(
            hidden_dims=mlp_hidden_dims,
            use_batchnorm=mlp_batchnorm,
            dropout=mlp_dropout,
        )

        self.use_class_bias = bool(use_class_bias)
        if self.use_class_bias:
            self.class_bias = nn.Parameter(torch.zeros(self.C))

        self.register_buffer("A_hat", torch.eye(self.C, dtype=torch.float32))
        self.register_buffer("gcn_do_mask", torch.ones(self.C, dtype=torch.bool))

        self.to(self.device)

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError(f"{name} must be > 0, got {value}.")
        return ivalue

    @staticmethod
    def _validate_unit_interval(value: float, name: str) -> float:
        fvalue = float(value)
        if not 0.0 <= fvalue <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
        return fvalue

    @staticmethod
    def _validate_non_negative(value: float, name: str) -> float:
        fvalue = float(value)
        if fvalue < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
        return fvalue

    def _build_gcn_layers(self, hidden_dims: tuple[int, ...]) -> nn.ModuleList:
        dims = [self.d] + [self._validate_positive_int(x, "gcn_hidden_dims") for x in hidden_dims] + [self.F]
        layers = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.kaiming_uniform_(layer.weight, a=self.neg_slope)
            layers.append(layer)
        return layers

    def _build_feature_mlp(
        self,
        hidden_dims: tuple[int, ...],
        use_batchnorm: bool,
        dropout: float,
    ) -> nn.Sequential:
        dropout = self._validate_unit_interval(dropout, "mlp_dropout")
        validated_hidden = [self._validate_positive_int(x, "mlp_hidden_dims") for x in hidden_dims]

        dims = [self.input_dim] + validated_hidden + [self.F]
        layers: list[nn.Module] = []

        for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(dims) - 2
            if is_last:
                continue
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(negative_slope=self.neg_slope))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        return nn.Sequential(*layers)

    @staticmethod
    def _row_normalize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rowsum = A.sum(dim=1, keepdim=True)
        return A / (rowsum + eps)

    def _label_embeddings(self) -> torch.Tensor:
        return self.Z

    @torch.no_grad()
    def build_label_graph(self, Y: NDArray[np.number]) -> torch.Tensor:
        """Build a directed row-normalized label graph from binary labels.

        Steps:
        1. Compute co-occurrence matrix ``M = Y.T @ Y``.
        2. Estimate ``P(Lj|Li)`` with Laplacian smoothing ``alpha``.
        3. Threshold by ``tau`` to obtain adjacency support.
        4. Allocate ``1-p`` to self-loop and ``p`` across outgoing neighbors.
        5. Row-normalize to produce ``A_hat``.
        """
        Y_np = np.asarray(Y)
        if Y_np.ndim != 2 or Y_np.shape[1] != self.C:
            raise ValueError(
                f"Y must have shape (N, {self.C}), got {tuple(Y_np.shape)}."
            )

        Yb = (Y_np > 0).astype(np.int64)
        M = Yb.T @ Yb

        label_counts = M.diagonal().astype(np.float64, copy=True)
        label_counts[label_counts == 0] = 1.0

        denom = label_counts[:, None] + self.alpha * self.C
        P = (M + self.alpha) / denom

        A = (P >= self.tau).astype(np.float32)
        A_prime = np.zeros_like(A, dtype=np.float32)

        for i in range(self.C):
            mask = A[i].astype(bool)
            mask[i] = False
            degree = int(mask.sum())

            if degree == 0:
                A_prime[i, i] = 1.0
                continue

            if self.p <= 0.0:
                A_prime[i, i] = 1.0
                continue

            A_prime[i, i] = max(0.0, 1.0 - self.p)
            A_prime[i, mask] = self.p / degree

        A_prime_t = torch.tensor(A_prime, dtype=torch.float32, device=self.device)

        # Ensure at least one non-self outgoing neighbor when possible.
        Atmp = A_prime_t.clone()
        Atmp.fill_diagonal_(0)
        lonely_nodes = ((Atmp > 0).sum(dim=1) == 0).nonzero(as_tuple=False).flatten()
        if lonely_nodes.numel() > 0:
            P_t = torch.tensor(P, dtype=torch.float32, device=self.device)
            for i in lonely_nodes.tolist():
                probs = P_t[i].clone()
                probs[i] = -1.0
                j = int(torch.argmax(probs).item())
                if probs[j] > 0:
                    A_prime_t[i, j] = max(A_prime_t[i, j].item(), min(1.0, self.p))

        self.A_hat = self._row_normalize(A_prime_t)
        stats = self._graph_stats(self.A_hat)
        self._apply_adaptive_policy(stats)
        return self.A_hat

    def _graph_stats(self, A: torch.Tensor) -> GraphStats:
        """Compute basic structural statistics for a dense or sparse graph."""
        C = int(A.size(0))
        with torch.no_grad():
            A_nodiag = A.to_dense().clone() if A.is_sparse else A.clone()
            A_nodiag.fill_diagonal_(0)
            E = int((A_nodiag > 0).sum().item())
            Emax = C * (C - 1)
            density = E / Emax if Emax > 0 else 0.0
            degrees = (A_nodiag > 0).sum(dim=1)
            mean_degree = float(degrees.float().mean().item()) if C > 0 else 0.0

        return {
            "density": float(density),
            "mean_degree": mean_degree,
            "degrees": degrees.cpu(),
            "E": E,
            "C": C,
        }

    def _apply_adaptive_policy(self, stats: GraphStats) -> None:
        """Light adaptive policy based on graph sparsity."""
        density = stats["density"]
        mean_degree = stats["mean_degree"]
        degrees = stats["degrees"]

        if mean_degree < 3 or density < 0.15:
            self.gcn_do_p = 0.0
        elif density < 0.35:
            self.gcn_do_p = min(self.gcn_do_p, 0.1)
        else:
            self.gcn_do_p = max(self.gcn_do_p, 0.2)

        self.gcn_do_mask = (degrees >= 3).to(self.device)

    def classifier_matrix(self) -> torch.Tensor:
        """Compute class-wise classifier matrix ``W`` with GCN propagation."""
        H = self._label_embeddings()
        A = self.A_hat

        for layer_idx, layer in enumerate(self.gcn_layers):
            H = A @ (H @ layer.weight.T)
            if layer_idx < len(self.gcn_layers) - 1:
                H = self.act(H)
                if self.training and self.gcn_do_p > 0.0:
                    if self.gcn_do_mask.any():
                        masked = F.dropout(H[self.gcn_do_mask], p=self.gcn_do_p, training=True)
                        H = H.clone()
                        H[self.gcn_do_mask] = masked
                    else:
                        H = F.dropout(H, p=self.gcn_do_p, training=True)
        return H

    def features(self, X: torch.Tensor) -> torch.Tensor:
        return self.feature_mlp(X)

    def logits(self, X: torch.Tensor) -> torch.Tensor:
        Phi = self.features(X)
        W = self.classifier_matrix()
        logits = Phi @ W.t()
        if self.use_class_bias:
            logits = logits + self.class_bias
        return logits

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.logits(X)

    def fit(
        self,
        X: NDArray[np.number],
        Y: NDArray[np.number],
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        verbose: bool = True,
        val: tuple[NDArray[np.number], NDArray[np.number]] | None = None,
        pos_weight: NDArray[np.number] | None = None,
        seed: int = 42,
    ) -> "LaRMLGCNClassifier":
        """Train end-to-end with BCEWithLogitsLoss."""
        X_np = np.asarray(X)
        Y_np = np.asarray(Y)

        if X_np.ndim != 2 or X_np.shape[1] != self.input_dim:
            raise ValueError(
                f"X must have shape (N, {self.input_dim}), got {tuple(X_np.shape)}."
            )
        if Y_np.ndim != 2 or Y_np.shape[1] != self.C:
            raise ValueError(
                f"Y must have shape (N, {self.C}), got {tuple(Y_np.shape)}."
            )
        if X_np.shape[0] != Y_np.shape[0]:
            raise ValueError("X and Y must have the same number of rows.")

        epochs = self._validate_positive_int(epochs, "epochs")
        batch_size = self._validate_positive_int(batch_size, "batch_size")
        lr = self._validate_non_negative(lr, "lr")
        weight_decay = self._validate_non_negative(weight_decay, "weight_decay")

        torch.manual_seed(seed)
        np.random.seed(seed)

        use_amp = torch.cuda.is_available()
        grad_clip = 1.0

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y_np, dtype=torch.float32, device=self.device)

        self.build_label_graph(Y_np)

        if self.use_class_bias:
            with torch.no_grad():
                pi = Y_t.mean(dim=0).clamp_(1e-6, 1 - 1e-6)
                if pos_weight is None:
                    init_prob = pi
                else:
                    posw_t = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
                    init_prob = (posw_t * pi) / (posw_t * pi + (1 - pi))
                    init_prob = init_prob.clamp_(1e-6, 1 - 1e-6)
                self.class_bias.copy_(torch.logit(init_prob))

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 10))

        criterion = (
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=self.device))
            if pos_weight is not None
            else nn.BCEWithLogitsLoss()
        )

        scaler: torch.amp.GradScaler | None
        if use_amp:
            try:
                scaler = torch.amp.GradScaler(device_type="cuda", enabled=True)
            except TypeError:
                scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = None

        N = X_t.size(0)
        indices = torch.arange(N, device=self.device)

        for epoch in range(1, epochs + 1):
            self.train()
            perm = indices[torch.randperm(N)]
            total_loss = 0.0

            for start in range(0, N, batch_size):
                sel = perm[start : start + batch_size]
                xb, yb = X_t[sel], Y_t[sel]
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    batch_logits = self(xb)
                    loss = criterion(batch_logits, yb)

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                    optimizer.step()

                total_loss += float(loss.item()) * xb.size(0)

            scheduler.step()

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch in (1, epochs)):
                message = f"[{epoch:03d}/{epochs}] loss={total_loss / N:.4f}"
                if val is not None:
                    try:
                        from sklearn.metrics import roc_auc_score

                        val_prob = self.predict_proba(val[0]).reshape(-1)
                        val_y = np.asarray(val[1]).reshape(-1)
                        if len(np.unique(val_y)) > 1:
                            auroc = roc_auc_score(val_y, val_prob)
                            message += f" | val AUROC(micro)={auroc:.4f}"
                    except Exception:
                        pass
                print(message)

        return self

    def _to_device_with_dtype(self, t: torch.Tensor) -> torch.Tensor:
        model_dtype = next(self.parameters()).dtype
        return t.to(self.device, dtype=model_dtype, non_blocking=True)

    @torch.inference_mode()
    def predict_proba(self, X: NDArray[np.number] | torch.Tensor) -> NDArray[np.float32]:
        self.eval()
        tensor = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        if tensor.ndim != 2 or tensor.shape[1] != self.input_dim:
            raise ValueError(
                f"X must have shape (N, {self.input_dim}), got {tuple(tensor.shape)}."
            )
        if self.device.type == "cuda" and tensor.device.type == "cpu":
            tensor = tensor.pin_memory()

        x = self._to_device_with_dtype(tensor)
        probs = torch.sigmoid(self(x)).float()
        return probs.cpu().numpy()

    @torch.inference_mode()
    def predict(
        self,
        X: NDArray[np.number] | torch.Tensor,
        threshold: float = 0.5,
        out_dtype=np.int8,
    ) -> NDArray[np.generic]:
        thr = self._validate_unit_interval(threshold, "threshold")
        proba = self.predict_proba(X)
        return (proba >= thr).astype(out_dtype, copy=False)


# Backward-compatible alias for existing code.
MLGCNClassifier = LaRMLGCNClassifier

