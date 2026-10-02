from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter, softmax, to_undirected

from monroe.model.constants import BOND_TYPE_FEAT_IDX, BOND_TYPE_OTHER_CODE, NODE_FLOAT_MISSING_IDXS


class MultiHeadAttentionLayerGritSparse(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        use_bias: bool,
        clamp: float = 5.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
        nn.init.xavier_normal_(self.VeRow)
        self.scale = self.out_dim ** -0.5

    def propagate_attention(self, batch: Data, attn_bias: torch.Tensor) -> Data:
        src = batch.K_h[batch.edge_index[0]]  # [E,H,D]
        dst = batch.Q_h[batch.edge_index[1]]  # [E,H,D]
        score = src * dst  # elementwise

        batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
        E_w, E_b = batch.E[:, :, : self.out_dim], batch.E[:, :, self.out_dim :]

        score = score * E_w
        score = torch.sign(score) * torch.sqrt(score.abs() + 1e-9)
        if self.clamp is not None:
            score = torch.clamp(score, -self.clamp, self.clamp)
        score = score + E_b

        e_t = score
        batch.wE = score.flatten(1)

        logits = torch.einsum("ehd,dhc->ehc", score.contiguous(), self.Aw.contiguous())
        logits = logits * self.scale + attn_bias
        if self.clamp is not None:
            logits = torch.clamp(logits, -self.clamp, self.clamp)
        alpha = softmax(logits, batch.edge_index[1])  # [E,H,1]

        batch.attn = self.dropout(alpha)

        msg = (batch.V_h[batch.edge_index[0]] * batch.attn).contiguous()  # [E,H,D]
        batch.wV = scatter(msg, batch.edge_index[1], dim=0, dim_size=batch.num_nodes, reduce="sum")

        rowV = scatter(
            e_t * batch.attn, batch.edge_index[1], dim=0, dim_size=batch.num_nodes, reduce="sum"
        ).contiguous()
        rowV = torch.einsum("nhd,dhc->nhc", rowV.contiguous(), self.VeRow.contiguous())
        batch.wV = batch.wV + rowV
        return batch

    def forward(self, batch: Data, attn_bias: Optional[torch.Tensor] = None):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        V_h = self.V(batch.x)
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        batch.E = self.E(batch.edge_attr)
        batch = self.propagate_attention(batch, attn_bias=attn_bias)
        return batch.wV, batch.wE


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head gated attention pooling:
      - per-node scores per head
      - per-graph softmax (segment softmax) per head
      - optional per-head value projection (standard MHA style: head_dim = in_dim // num_heads)
      - weighted sum per head
      - concatenate heads -> in_dim (when per_head_values=True)
      - or head-averaged output (keeps dim = in_dim, when per_head_values=False)
    """
    def __init__(
        self,
        in_dim: int,
        num_heads: int = 8,
        gate_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.gate_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else nn.Identity()

        self.gate = nn.Linear(in_dim, num_heads, bias=True)

        self.head_dim = in_dim // num_heads
        self.value_proj = nn.Linear(in_dim, in_dim, bias=False)
        nn.init.xavier_uniform_(self.value_proj.weight)


    def forward(self, batch):
        x = batch.x                    # [N, d]
        b = batch.batch                # [N]
        num_nodes = batch.num_nodes    # total N 

        # scores: [N, H]
        scores = self.gate_dropout(self.gate(x))

        # segment softmax per head: [N, H]
        alpha = softmax(scores, b, num_nodes=num_nodes)

        v = self.value_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        weighted = v * alpha.unsqueeze(-1)
        out = scatter(weighted, b, dim=0, reduce="sum")
        out = out.flatten(1)

        return out


class GritTransformerLayer(nn.Module):
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        clamp: float = 5.0,
        bn_momentum: float = 0.1):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            dropout=attn_dropout,
            clamp=clamp,
            use_bias=True,
        )

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)
        self.deg_coef = nn.Parameter(torch.zeros(1, out_dim, 2))
        nn.init.xavier_normal_(self.deg_coef)

        self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=True, eps=1e-5, momentum=bn_momentum)
        self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=True, eps=1e-5, momentum=bn_momentum)
        self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=True, eps=1e-5, momentum=bn_momentum)

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

    def _post_attn_ffn(self, h_attn, e_attn, h_in, e_in, log_deg):
        """Dense post-attention projection, residual connections, BN, and FFN."""
        h = h_attn.flatten(1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        h = torch.stack([h, h * log_deg], dim=-1)
        h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        e = e_attn.flatten(1)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.O_e(e)

        h = h_in + h  # residual connection
        e = e + e_in

        h = self.batch_norm1_h(h)
        e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        h = h_in2 + h  # residual connection
        h = self.batch_norm2_h(h)

        return h, e

    def forward(self, batch: Data) -> Data:
        log_deg = batch.log_deg
        if log_deg.dim() == 1:
            log_deg = log_deg.unsqueeze(-1)

        h_attn_out, e_attn_out = self.attention(batch, attn_bias=batch.rwpe_attn_bias)
        h, e = self._post_attn_ffn(h_attn_out, e_attn_out, batch.x, batch.edge_attr, log_deg)

        batch.x = h
        batch.edge_attr = e

        return batch

    def __repr__(self):
        return "{}(in_dim={}, out_dim={}, heads={})\n[{}]".format(
            self.__class__.__name__,
            self.in_dim,
            self.out_dim,
            self.num_heads,
            super().__repr__(),
        )


class RBF(nn.Module):
    def __init__(self, rbf_dim, d_min=0.0, d_max=5.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, rbf_dim)
        delta = (d_max - d_min) / max(rbf_dim - 1, 1)
        gamma = 0.5 / (delta * delta)
        self.register_buffer("centers", centers)
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:  # (E,1) -> (E,rbf_dim)
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers)**2)


NODE_FLOAT_RBF_RANGES = [
    (0.0, 8.0),       # num_radical_electrons
    (0.5, 4.0),       # electronegativity
    (3.0, 25.0),      # first_ionization (eV)
    (0.2, 2.7),       # covalent_radius (A)
    (0.0, 4000.0),    # melting_point (K)
    (-8.0, 8.0),      # formal_charge
    (1.0, 3.0),       # vdw_radius (A)
    (0.0, 300.0),     # atomic_mass
]


class PerFeatureRBF(nn.Module):
    """RBF expansion applied independently per scalar feature. [N, F] -> [N, F*R]."""

    def __init__(self, n_features: int, rbf_dim: int, ranges: list):
        super().__init__()
        all_centers, all_gammas = [], []
        for d_min, d_max in ranges:
            all_centers.append(torch.linspace(d_min, d_max, rbf_dim))
            delta = (d_max - d_min) / max(rbf_dim - 1, 1)
            all_gammas.append(0.5 / (delta * delta))
        self.register_buffer("centers", torch.stack(all_centers))  # [F, R]
        self.register_buffer("gammas", torch.tensor(all_gammas))   # [F]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, F] -> [N, F*R]
        diff = x.unsqueeze(-1) - self.centers.unsqueeze(0)         # [N, F, R]
        return torch.exp(-self.gammas[None, :, None] * diff * diff).flatten(1)


class GritTransformer(nn.Module):
    def __init__(
        self,
        node_feature_vocab: Dict[str, Sequence],
        edge_feature_vocab: Dict[str, Sequence],
        node_float_dim: int,
        rbf_dim: int,
        walk_len: int,
        emb_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        clamp: float = 5.0,
        bn_momentum: float = 0.1,
        structure_prediction: bool = False,
        use_stereo_edges: bool = True,
        zero_vn_edge_rbf: bool = False,
    ):
        super().__init__()

        self.structure_prediction = structure_prediction
        self.use_stereo_edges = use_stereo_edges
        self.zero_vn_edge_rbf = zero_vn_edge_rbf
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.node_float_dim = node_float_dim
        self.rbf_dim = rbf_dim
        self.edge_rbf = RBF(rbf_dim, d_min=0.5, d_max=5.0)
        self.walk_len = walk_len

        self.missing_fill = nn.Parameter(torch.zeros(len(NODE_FLOAT_MISSING_IDXS)))
        self.node_float_rbf = PerFeatureRBF(node_float_dim, rbf_dim, NODE_FLOAT_RBF_RANGES)

        if self.rbf_dim < 0:
            raise ValueError("rbf_dim must be non-negative")
        if self.walk_len < 3:
            raise ValueError("walk_len must be at least 3 to compute rrwp features")

        self.node_feat_keys = list(node_feature_vocab.keys())
        self.edge_feat_keys = list(edge_feature_vocab.keys())

        self.node_oh = nn.ModuleDict({
            k: nn.Embedding(len(option_list) + 1, emb_dim)
            for k, option_list in node_feature_vocab.items()
        })
        self.edge_oh = nn.ModuleDict({
            k: nn.Embedding(len(option_list) + 1, emb_dim)
            for k, option_list in edge_feature_vocab.items()
        })

        for emb in list(self.node_oh.values()) + list(self.edge_oh.values()):
            nn.init.xavier_uniform_(emb.weight)

        node_in_dim = emb_dim + node_float_dim * rbf_dim + self.walk_len
        if self.rbf_dim > 0:
            edge_in_dim = emb_dim + self.rbf_dim + self.walk_len
        else:
            edge_in_dim = emb_dim + self.walk_len

        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        self.rwpe_attn_bias = nn.Linear(self.walk_len, num_heads, bias=False)

        self.grit_layers = nn.ModuleList([
            GritTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                clamp=clamp,
                bn_momentum=bn_momentum,
            ) for _ in range(num_layers)
        ])

        self.pooling = MultiHeadAttentionPooling(
            in_dim=hidden_dim,
            num_heads=num_heads,
            gate_dropout=dropout,
        )

    def _add_virtual_node(
        self,
        batch: Data,
        rrwp_node: torch.Tensor,
        rrwp_edge: torch.Tensor,
        log_deg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_batch = batch.batch
        device = batch.edge_index.device
        n_graphs = int(node_batch.max()) + 1
        n_nodes = batch.num_nodes
        vn_idx = torch.arange(n_graphs, device=device) + n_nodes

        node_codes_vn = torch.stack([
            torch.full(
                (n_graphs,),
                self.node_oh[k].num_embeddings - 1,
                device=batch.node_codes.device,
                dtype=batch.node_codes.dtype,
            )
            for k in self.node_feat_keys
        ], dim=1)

        batch.node_codes = torch.cat([batch.node_codes, node_codes_vn], dim=0)
        batch.x = torch.cat([
            batch.x,
            torch.zeros(n_graphs, self.node_float_dim, device=batch.x.device, dtype=batch.x.dtype),
        ], dim=0)
        batch.pos_in = torch.cat([
            batch.pos_in,
            torch.zeros(n_graphs, batch.pos_in.size(1), device=batch.pos_in.device, dtype=batch.pos_in.dtype),
        ], dim=0)
        batch.batch = torch.cat([
            node_batch,
            torch.arange(n_graphs, device=node_batch.device, dtype=node_batch.dtype),
        ], dim=0)

        nodes = torch.arange(n_nodes, device=device)
        vn_targets = n_nodes + node_batch
        edge_vn_fwd = torch.stack([nodes, vn_targets], dim=0)
        edge_vn_rev = edge_vn_fwd.flip(0)
        edge_vn = torch.cat([edge_vn_fwd, edge_vn_rev], dim=1)
        edge_codes_vn = torch.stack([
            torch.full(
                (n_nodes,),
                self.edge_oh[k].num_embeddings - 1,
                device=batch.edge_codes.device,
                dtype=batch.edge_codes.dtype,
            )
            for k in self.edge_feat_keys
        ], dim=1)
        edge_codes_vn = edge_codes_vn.repeat(2, 1)  # both directions

        batch.edge_index = torch.cat([batch.edge_index, edge_vn], dim=1)
        batch.edge_codes = torch.cat([batch.edge_codes, edge_codes_vn], dim=0)

        rrwp_node = torch.cat([
            rrwp_node,
            rrwp_node.new_zeros((n_graphs, self.walk_len)),
        ], dim=0)
        rrwp_edge = torch.cat([
            rrwp_edge,
            rrwp_edge.new_zeros((edge_vn.size(1), self.walk_len)),
        ], dim=0)
        vn_log_deg = torch.zeros(
            (n_graphs,) + log_deg.shape[1:],
            device=log_deg.device,
            dtype=log_deg.dtype,
        )
        log_deg = torch.cat([log_deg, vn_log_deg], dim=0)

        batch.num_nodes = batch.node_codes.size(0)
        batch.virtual_node_index = vn_idx
        return rrwp_node, rrwp_edge, log_deg

    def _get_rrwp(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        scope: str = "bonds",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        assert scope in {"all", "bonds"}, "scope must be 'all' or 'bonds'"

        ei_rw = to_undirected(edge_index, num_nodes=num_nodes)

        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float, device=ei_rw.device)
        adj[ei_rw[0], ei_rw[1]] = 1.0
        adj.fill_diagonal_(1.0)
        deg = adj.sum(dim=1)
        transition = adj / deg.unsqueeze(1)

        powers = []
        current = transition
        for _ in range(self.walk_len):
            powers.append(current)
            current = current @ transition
        probs = torch.stack(powers, dim=-1)

        rrwp_node_val = probs.diagonal(dim1=0, dim2=1).transpose(0, 1)

        if scope == "bonds":
            rows, cols = edge_index
            edge_vals = probs[rows, cols, :]
            edge_vals = 0.5 * (edge_vals + probs[cols, rows, :])
            rrwp_edge_index = edge_index
            rrwp_edge_val = edge_vals.contiguous()
        else:
            raise NotImplementedError("scope='all' is not supported.")

        log_deg = torch.log(deg + 1)
        return rrwp_node_val, rrwp_edge_index, rrwp_edge_val, log_deg

    def forward(
        self, 
        batch: Data, 
        node_idxs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        edge_index = batch.edge_index
        edge_codes = batch.edge_codes
        pos = batch.pos_in
        node_float = batch.x
        node_codes = batch.node_codes
        N = batch.num_nodes
        batch.num_real_nodes = N

        if not self.use_stereo_edges:
            is_real_bond = edge_codes[:, BOND_TYPE_FEAT_IDX] != BOND_TYPE_OTHER_CODE
            edge_index = edge_index[:, is_real_bond]
            edge_codes = edge_codes[is_real_bond]
            batch.edge_index = edge_index
            batch.edge_codes = edge_codes
            # Also filter precomputed RRWP edges if present
            if getattr(batch, "rrwp_edges", None) is not None:
                batch.rrwp_edges = batch.rrwp_edges[is_real_bond]

        if getattr(batch, "rrwp_nodes", None) is not None:
            rrwp_node = batch.rrwp_nodes
            rrwp_edge = batch.rrwp_edges
            log_deg = batch.log_deg
            if rrwp_node.shape[-1] != self.walk_len:
                raise ValueError(
                    f"Precomputed RRWP has walk_len={rrwp_node.shape[-1]} but this "
                    f"encoder was built with walk_len={self.walk_len}. Re-run RRWP "
                    f"precomputation with --walk-len {self.walk_len} (training data), "
                    f"or delete the stale eval RRWP cache so it recomputes."
                )
        else: # compute on the fly
            # this is slow but it's ok because precomputed for pretraining
            # and only used for downstream inference
            E = edge_index.size(1)
            if E == 0 or edge_codes.dim() < 2:
                # No edges: RRWP is just self-loop random walks
                rrwp_node, _, rrwp_edge, log_deg = self._get_rrwp(
                    N, edge_index.new_empty(2, 0)
                )
            else:
                # Build transition matrix from real bonds only (exclude stereo edges),
                # matching precompute_rrwp.py. Then index ALL edges (including stereo)
                # into the probability matrix, so stereo edges get their actual walk
                # probabilities — matching the precomputed training behavior.
                is_real_bond = edge_codes[:, BOND_TYPE_FEAT_IDX] != BOND_TYPE_OTHER_CODE
                real_edge_index = edge_index[:, is_real_bond]
                rrwp_node, _, _, log_deg = self._get_rrwp(N, real_edge_index)
                # Recompute edge RRWP for ALL edges using the real-bond transition matrix
                ei_rw = to_undirected(real_edge_index, num_nodes=N)
                adj = torch.zeros(N, N, dtype=torch.float, device=edge_index.device)
                adj[ei_rw[0], ei_rw[1]] = 1.0
                adj.fill_diagonal_(1.0)
                deg = adj.sum(dim=1)
                transition = adj / deg.unsqueeze(1)
                powers = []
                current = transition
                for _ in range(self.walk_len):
                    powers.append(current)
                    current = current @ transition
                probs = torch.stack(powers, dim=-1)
                rows, cols = edge_index
                edge_vals = probs[rows, cols, :]
                rrwp_edge = 0.5 * (edge_vals + probs[cols, rows, :])

        n_real_edges = batch.edge_index.size(1)
        rrwp_node, rrwp_edge, log_deg = self._add_virtual_node(batch, rrwp_node, rrwp_edge, log_deg)
        edge_index, edge_codes, pos, node_float, node_codes = (
            batch.edge_index,
            batch.edge_codes,
            batch.pos_in,
            batch.x,
            batch.node_codes,
        )
        N = batch.num_nodes

        # sum embedded one-hots (stack and sum for efficiency)
        emb_node = torch.stack(
            [self.node_oh[k](node_codes[:, idx]) for idx, k in enumerate(self.node_feat_keys)],
            dim=0,
        ).sum(dim=0)

        emb_edge = torch.stack(
            [self.edge_oh[k](edge_codes[:, idx]) for idx, k in enumerate(self.edge_feat_keys)],
            dim=0,
        ).sum(dim=0)

        src, dst = edge_index
        bond_vec = pos[src] - pos[dst]
        bond_lengths = bond_vec.norm(dim=-1)
        edge_rbf = self.edge_rbf(bond_lengths)
        if self.zero_vn_edge_rbf:
            # Virtual node edges (appended by _add_virtual_node) have
            # frame-dependent distances (||atom_pos - origin||). Zero them
            # out so the model relies on the VN's OOV edge code embedding
            # and zero RRWP instead of arbitrary geometric distances.
            edge_rbf[n_real_edges:] = 0.0
        e_cat = torch.cat([emb_edge, edge_rbf, rrwp_edge], dim=-1)      # [E, emb+R_e+W]

        # Replace 0.0 missing-value sentinels with learned fill values
        node_float = node_float.clone()
        fill = self.missing_fill.to(node_float.dtype)
        # Only fill real nodes — virtual nodes should keep their all-zeros
        real_mask = torch.ones(node_float.size(0), dtype=torch.bool, device=node_float.device)
        real_mask[batch.virtual_node_index] = False
        for i, col in enumerate(NODE_FLOAT_MISSING_IDXS):
            mask = real_mask & (node_float[:, col] == 0.0)
            node_float[mask, col] = fill[i]

        node_float_enc = self.node_float_rbf(node_float).clone()
        # Zero out RBF activations for virtual nodes — their all-zeros input
        # would otherwise produce spurious activations for features with d_min=0
        node_float_enc[batch.virtual_node_index] = 0.0
        x_cat = torch.cat([emb_node, node_float_enc, rrwp_node], dim=-1)

        x = self.node_proj(x_cat)                                       # [N, H]
        e = self.edge_proj(e_cat)                                       # [E, H]

        batch.x = x
        batch.edge_attr = e
        batch.log_deg = log_deg
        batch.rwpe_attn_bias = self.rwpe_attn_bias(rrwp_edge).unsqueeze(-1)  # [E, heads, 1]

        for layer in self.grit_layers:
            batch = layer(batch)

        node_level = None
        if node_idxs is not None:
            node_level = batch.x[node_idxs]

        graph_level = self.pooling(batch)

        return graph_level, node_level
