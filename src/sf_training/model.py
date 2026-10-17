"""pi0.5 flow matching with optional Spatial Forcing, sharing one policy path.

Uses the pinned Spatial-Forcing OpenPI fork (Apache-2.0) and VGGT backbone.
Geometry-changing augmentation is disabled identically in both arms so that
student and teacher patches refer to the same pixels. Color jitter is student-only.
"""

from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_align_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models_pytorch.projectors import AlignProjector
from vggt.heads.utils import custom_pooling
from sf_training.lora import inject_lora


def observation(batch, augment):
    images = batch["images"]  # B,2,3,224,224, RGB [0,1]
    teacher_images = images
    if augment:
        # Same RNG use and augmentation for baseline/SF; no teacher RNG calls.
        views = []
        for view in images.unbind(1):
            b = 0.7 + torch.rand((), device=view.device) * 0.6
            c = 0.6 + torch.rand((), device=view.device) * 0.8
            s = 0.5 + torch.rand((), device=view.device)
            x = view * b
            mean = x.mean((1, 2, 3), keepdim=True)
            x = (x - mean) * c + mean
            gray = x.mean(1, keepdim=True)
            views.append((gray + (x - gray) * s).clamp(0, 1))
        images = torch.stack(views, dim=1)
    b = images.shape[0]
    device = images.device
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    imgs = [images[:, 0] * 2 - 1, images[:, 1] * 2 - 1, torch.zeros_like(images[:, 0])]
    masks = [torch.ones(b, dtype=torch.bool, device=device)] * 2 + [
        torch.zeros(b, dtype=torch.bool, device=device)
    ]
    return SimpleNamespace(
        images=dict(zip(keys, imgs)),
        image_masks=dict(zip(keys, masks)),
        state=batch["state"],
        tokenized_prompt=batch["tokens"],
        tokenized_prompt_mask=batch["token_mask"],
    ), teacher_images


class Policy(PI0Pytorch):
    def __init__(self, cfg):
        sf = cfg["sf"]
        super().__init__(
            Pi0Config(
                pi05=True,
                action_horizon=cfg["action_horizon"],
                discrete_state_input=False,
                max_token_len=cfg["max_token_len"],
            ),
            SimpleNamespace(
                vla_layers_align=sf["student_layer_index"],
                vggt_layers_align=sf["teacher_layer_index"],
                pooling_func="bilinear",
                use_vggt_pe=sf["position_embedding"],
            ),
        )
        # Avoid inference compilation during training; preserve the upstream method.
        self.sample_actions = PI0Pytorch.sample_actions.__get__(self, Policy)

    def _preprocess_observation(self, obs, *, train=True, get_wo_aug=False):
        return (
            list(obs.images.values()),
            None,
            None,
            list(obs.image_masks.values()),
            obs.tokenized_prompt,
            obs.tokenized_prompt_mask,
            obs.state,
        )

    def flow(self, obs, actions, need_features):
        images, _, _, masks, tokens, token_masks, state = self._preprocess_observation(
            obs
        )
        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)
        t = time[:, None, None]
        noisy = t * noise + (1 - t) * actions
        target = noise - actions
        prefix, pm, pa, img_len = self.embed_prefix(images, masks, tokens, token_masks)
        suffix, sm, sa, cond = self.embed_suffix(state, noisy, time)
        prefix = prefix.to(torch.bfloat16)
        suffix = suffix.to(torch.bfloat16)
        padding = torch.cat([pm, sm], 1)
        attention = torch.cat([pa, sa], 1)
        positions = torch.cumsum(padding, 1) - 1
        mask4 = self._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
        result = self.paligemma_with_expert(
            attention_mask=mask4,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            adarms_cond=[None, cond],
            output_hidden_states=need_features,
        )
        predicted = self.action_out_proj(
            result[0][1][:, -self.config.action_horizon :].float()
        ).float()
        # Match OpenPI: average all 32 padded action coordinates, not just 7 robot DOFs.
        fm = F.mse_loss(predicted, target)
        features = (
            result[2][self.vla_layers_align][0][:, : img_len * 2 // 3]
            if need_features
            else None
        )
        return fm, features


class Experiment(nn.Module):
    def __init__(self, policy, cfg, arm, teacher=None):
        super().__init__()
        self.policy = policy
        self.cfg = cfg
        self.arm = arm
        self.projector = (
            AlignProjector(2048, 1024, use_vlm_norm=False) if arm == "sf" else None
        )
        # Intentionally not registered: frozen teacher is excluded from DDP, optimizer and checkpoints.
        object.__setattr__(self, "teacher", teacher)
        if arm == "sf" and teacher is None:
            raise ValueError("SF requires a frozen VGGT teacher")

    def forward(self, batch):
        obs, teacher_images = observation(batch, augment=self.training)
        fm, features = self.policy.flow(
            obs, batch["actions"], need_features=self.arm == "sf"
        )
        align = fm.new_zeros(())
        if self.arm == "sf":
            with torch.no_grad():
                images = F.interpolate(
                    teacher_images.flatten(0, 1),
                    size=(518, 518),
                    mode="bicubic",
                    align_corners=False,
                ).unflatten(0, (teacher_images.shape[0], 2))
                # Only real cameras enter VGGT; the masked third OpenPI slot is excluded.
                output = self.teacher(images.clamp(0, 1))
                hidden = output["features"][self.cfg["sf"]["teacher_layer_index"]][
                    :, :, output["patch_start_idx"] :, :
                ]
                target = custom_pooling(
                    hidden,
                    (37, 37),
                    (518, 518),
                    features,
                    "bilinear",
                    self.cfg["sf"]["position_embedding"],
                )
            mask = torch.ones(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
            align = self.projector(features, target, mask)
        return fm + self.cfg["sf"]["weight"] * align, fm.detach(), align.detach()


def build_policy(cfg, base_file):
    import safetensors.torch

    policy = Policy(cfg)
    # load_model resolves tied embedding aliases and raises on all missing/unexpected keys.
    safetensors.torch.load_model(policy, str(base_file), strict=True)
    adapters = inject_lora(policy, cfg["lora"])
    if len(adapters) != 252:
        raise RuntimeError(f"Expected 252 adapted projections, got {len(adapters)}")
    if cfg["gradient_checkpointing"]:
        policy.gradient_checkpointing_enable()
    return policy


def build_teacher(path):
    from vggt.models.vggt import VGGT

    teacher = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
        feature_only=True,
    )
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    # Explicitly omit prediction heads; every aggregator tensor must match strictly.
    teacher.aggregator.load_state_dict(
        {
            k.removeprefix("aggregator."): v
            for k, v in state.items()
            if k.startswith("aggregator.")
        },
        strict=True,
    )
    teacher.requires_grad_(False).eval()
    return teacher
