"""PyTorch implementation of Arena/OpenPI's per-head attention LoRA axes."""

import torch
from torch import nn


class ArenaLoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, kind="dense", heads=1, init_std=0.01):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.kind, self.heads, self.scale = kind, heads, alpha / rank
        i, o = base.in_features, base.out_features
        # JAX axes (-2,-1): independent rank-r update for each attention head.
        ashape = (
            (heads, i, rank)
            if kind == "qkv"
            else ((heads, i // heads, rank) if kind == "out" else (i, rank))
        )
        bshape = (
            (heads, rank, o // heads)
            if kind == "qkv"
            else ((heads, rank, o) if kind == "out" else (rank, o))
        )
        self.lora_a = nn.Parameter(
            torch.empty(ashape, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.empty(bshape, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.normal_(self.lora_a, std=init_std)
        nn.init.normal_(self.lora_b, std=init_std)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def delta(self, x):
        # autocast handles BF16 compute while adapters retain FP32 master weights.
        x = x.to(self.lora_a.dtype)
        if self.kind == "qkv":
            return torch.einsum(
                "...hr,hro->...ho",
                torch.einsum("...i,hir->...hr", x, self.lora_a),
                self.lora_b,
            ).flatten(-2)
        if self.kind == "out":
            x = x.unflatten(-1, (self.heads, -1))
            return torch.einsum(
                "...hr,hro->...o",
                torch.einsum("...hi,hir->...hr", x, self.lora_a),
                self.lora_b,
            )
        return x @ self.lora_a @ self.lora_b

    def forward(self, x):
        y = self.base(x)
        return y + self.delta(x).to(y.dtype) * self.scale

    def merged_weight(self):
        d = self.lora_a @ self.lora_b
        if self.kind == "qkv":
            d = d.permute(1, 0, 2).flatten(1)
        elif self.kind == "out":
            d = d.flatten(0, 1)
        return self.base.weight.float() + d.T * self.scale


def inject_lora(policy, cfg):
    backbone = policy.paligemma_with_expert
    # The original JAX filter freezes the entire LLM including expert adaRMS.
    backbone.paligemma.language_model.requires_grad_(False)
    backbone.paligemma.lm_head.requires_grad_(False)
    backbone.gemma_expert.requires_grad_(False)
    records = []
    for label, branch, rank, alpha in [
        ("vlm", backbone.paligemma.language_model, cfg["vlm_rank"], cfg["vlm_alpha"]),
        (
            "expert",
            backbone.gemma_expert.model,
            cfg["expert_rank"],
            cfg["expert_alpha"],
        ),
    ]:
        for layer_index, layer in enumerate(branch.layers):
            for group, names in [
                ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                ("mlp", ("gate_proj", "up_proj", "down_proj")),
            ]:
                parent = getattr(layer, group)
                for name in names:
                    base = getattr(parent, name)
                    heads = 8 if name in ("q_proj", "o_proj") else 1
                    kind = (
                        "out"
                        if name == "o_proj"
                        else ("qkv" if group == "self_attn" else "dense")
                    )
                    setattr(
                        parent,
                        name,
                        ArenaLoRALinear(
                            base, rank, alpha, kind, heads, cfg["init_std"]
                        ),
                    )
                    records.append(f"{label}.{layer_index}.{group}.{name}")
    # Full training of image encoder/projector and action/time projections, as in Arena.
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return records


def parameter_manifest(module):
    rows = [
        dict(
            name=n,
            shape=list(p.shape),
            numel=p.numel(),
            dtype=str(p.dtype),
            trainable=p.requires_grad,
        )
        for n, p in module.named_parameters()
    ]
    return dict(
        parameters=rows,
        total=sum(x["numel"] for x in rows),
        trainable=sum(x["numel"] for x in rows if x["trainable"]),
    )


def trainable_state(module):
    return {
        n: p.detach().cpu() for n, p in module.named_parameters() if p.requires_grad
    }


def load_trainable(module, state):
    wanted = {n for n, p in module.named_parameters() if p.requires_grad}
    if wanted != set(state):
        raise ValueError(
            f"Trainable checkpoint mismatch: {wanted.symmetric_difference(state)}"
        )
    with torch.no_grad():
        for n, p in module.named_parameters():
            if n in state:
                if p.shape != state[n].shape:
                    raise ValueError(f"Shape mismatch: {n}")
                p.copy_(state[n])
