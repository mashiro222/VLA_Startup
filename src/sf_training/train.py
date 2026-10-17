"""Single-node DDP trainer; full runs use identical global batch across GPU counts."""

import argparse
import contextlib
import json
import os
import time
import signal
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sf_training import checkpoint
from sf_training.data import ArenaData, batch_indices, collate
from sf_training.lora import parameter_manifest
from sf_training.model import Experiment, build_policy, build_teacher
from sf_training.utils import accumulation, learning_rate, seed_all, write_json, sha
from sf_training.normalization import MAX_FLOW_MSE, unnormalize

STOP = False


def stop_requested(*_):
    global STOP
    STOP = True


def run(plan_path, arm, stop_step=None, resume=False):
    from sf_training.workflow import verify_plan

    plan = verify_plan(Path(plan_path))
    cfg = plan["recipe"]
    root = Path(plan["root"])
    home = root / ".runtime/training"
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world != plan["gpus"]:
        raise ValueError("World size differs from prepared plan")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training must run inside a Slurm GPU allocation")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    accum = accumulation(cfg, world)
    seed_all(cfg["seed"])
    policy = build_policy(cfg, home / "pi05_base_torch/model.safetensors").to(device)
    # Paired policy/adapter initialization does not depend on the teacher or projector.
    seed_all(cfg["seed"] + 1000)
    teacher = build_teacher(home / "vggt/model.pt").to(device) if arm == "sf" else None
    model = Experiment(policy, cfg, arm, teacher).to(device).train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"]["peak"],
        betas=tuple(cfg["optimizer"]["betas"]),
        eps=cfg["optimizer"]["eps"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    output = Path(plan_path).parent / arm
    output.mkdir(exist_ok=True)
    identity = dict(
        plan_sha256=sha(plan_path),
        arm=arm,
        world=world,
        base=sha(home / "pi05_base_torch/conversion.json"),
    )
    start = 0
    ckpts = sorted(output.glob("step-*.pt"))
    seed_all(cfg["seed"] + rank)
    if resume:
        if not ckpts:
            raise FileNotFoundError("No complete checkpoint to resume")
        start = checkpoint.restore(ckpts[-1], model, optimizer, identity, rank)
    elif ckpts:
        raise FileExistsError("Run already contains checkpoints; use --resume")
    if rank == 0:
        write_json(output / "parameters.json", parameter_manifest(model))
        write_json(
            output / "runtime.json",
            dict(
                gpu=torch.cuda.get_device_name(),
                total_memory=torch.cuda.get_device_properties(device).total_memory,
                torch=torch.__version__,
                world=world,
                global_batch=cfg["global_batch"],
                micro_batch=cfg["micro_batch"],
                accumulation=accum,
                arm=arm,
            ),
        )
    stats = json.loads((home / "norm_stats.json").read_text())["norm_stats"]
    dataset = ArenaData(
        home / "arena_l0",
        root / ".runtime/cache/openpi/big_vision/paligemma_tokenizer.model",
        stats,
        cfg["action_horizon"],
        cfg["max_token_len"],
    )
    wrapped = (
        DDP(
            model,
            device_ids=[local],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        if world > 1
        else model
    )
    target = min(cfg["steps"], stop_step if stop_step is not None else cfg["steps"])
    signal.signal(signal.SIGUSR1, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    updates = 0
    last_step = start
    for step in range(start, target):
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, cfg["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = batch_indices(
            len(dataset), cfg["seed"], step, cfg["global_batch"], rank, world
        )
        losses = torch.zeros(3, device=device)
        target_abs_max = torch.zeros((), device=device)
        rare_rotation_samples = torch.zeros((), device=device)
        for micro in range(accum):
            subset = indices[
                micro * cfg["micro_batch"] : (micro + 1) * cfg["micro_batch"]
            ]
            batch = collate([dataset[i] for i in subset], device)
            target_abs_max = torch.maximum(target_abs_max, batch["actions"].abs().max())
            rare_rotation_samples += batch["rare_rotation"].sum()
            sync = (
                wrapped.no_sync()
                if world > 1 and micro < accum - 1
                else contextlib.nullcontext()
            )
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, fm, align = wrapped(batch)
                    if not torch.isfinite(loss) or not torch.isfinite(fm) or fm > MAX_FLOW_MSE:
                        raise FloatingPointError(f"Nonfinite/extreme flow loss at update {step + 1}: {float(fm)}")
                    scaled = loss / accum
                scaled.backward()
            losses += torch.stack([loss.detach(), fm, align]) / accum
        grads = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        if not grads or not all(torch.isfinite(p.grad).all() for _, p in grads):
            raise FloatingPointError("Missing/nonfinite gradients")
        if step == start:
            names = [n for n, _ in grads]
            for label, test in [
                ("LoRA", lambda n: "lora_" in n),
                ("vision", lambda n: "vision_tower" in n),
                ("action", lambda n: "action_out_proj" in n),
            ]:
                if not any(test(n) for n in names):
                    raise RuntimeError(f"Missing {label} gradient")
            if arm == "sf" and (
                not any(n.startswith("projector.") for n in names)
                or any(p.grad is not None for p in teacher.parameters())
            ):
                raise RuntimeError("SF gradient/frozen-teacher invariant failed")
            if rank == 0:
                write_json(
                    output / f"gradients-step-{step + 1}.json",
                    dict(
                        with_grad=names,
                        without_grad=[
                            n
                            for n, p in model.named_parameters()
                            if p.requires_grad and p.grad is None
                        ],
                    ),
                )
        gradnorm = torch.nn.utils.clip_grad_norm_(
            [p for _, p in grads],
            cfg["optimizer"]["clip_grad_norm"],
            error_if_nonfinite=True,
        )
        optimizer.step()
        last_step = step + 1
        updates += 1
        if world > 1:
            dist.all_reduce(losses)
            losses /= world
            dist.all_reduce(target_abs_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(rare_rotation_samples)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - begin
        if rank == 0:
            record = dict(
                step=last_step,
                loss=float(losses[0]),
                flow_mse=float(losses[1]),
                alignment=float(losses[2]),
                lr=lr,
                grad_norm=float(gradnorm),
                target_abs_max=float(target_abs_max),
                rare_rotation_samples=int(rare_rotation_samples),
                seconds=elapsed,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                samples_per_second=cfg["global_batch"] / elapsed,
            )
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if last_step % cfg["log_every"] == 0 or last_step <= 3:
                print(json.dumps(record), flush=True)
        stopping = torch.tensor(
            int(
                bool(
                    STOP
                    or (
                        os.environ.get("SF_DEADLINE")
                        and time.time() > float(os.environ["SF_DEADLINE"])
                    )
                )
            ),
            device=device,
        )
        if world > 1:
            dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
        if (
            last_step % cfg["checkpoint_every"] == 0
            or last_step == target
            or stopping.item()
        ):
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, checkpoint.rng_state())
            else:
                states = [checkpoint.rng_state()]
            if rank == 0:
                checkpoint.save(
                    output / f"step-{last_step:08d}.pt",
                    model,
                    optimizer,
                    last_step,
                    identity,
                    states,
                )
                for old in sorted(output.glob("step-*.pt"))[: -cfg["keep_checkpoints"]]:
                    old.unlink()
            if world > 1:
                dist.barrier()
        if stopping.item():
            break
    if plan["profile"] == "smoke" and last_step == cfg["steps"]:
        from sf_training.model import observation

        model.eval()
        sample = collate([dataset[0]], device)
        obs, _ = observation(sample, False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actions = model.policy.sample_actions(
                device, obs, noise=torch.zeros((1, 10, 32), device=device), num_steps=10
            )
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Inference produced nonfinite actions")
        raw_actions = unnormalize(actions[0, :, :7].float().cpu().numpy(), stats["actions"])
        if rank == 0:
            write_json(
                output / "inference_smoke.json",
                dict(
                    finite=True,
                    shape=list(actions.shape),
                    actions_first7=actions[0, :, :7].float().cpu().tolist(),
                    raw_actions_first7=raw_actions.tolist(),
                ),
            )
    if rank == 0:
        write_json(
            output / "status.json",
            dict(
                completed=last_step == cfg["steps"],
                step=last_step,
                target=cfg["steps"],
                updates_this_process=updates,
                seconds_this_process=time.monotonic() - started,
                interrupted=STOP,
                stop_reason="target_reached"
                if last_step == target
                else ("signal" if STOP else "wall_time_budget"),
                resumed_from=start,
            ),
        )
    if world > 1:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--arm", choices=["baseline", "sf"], required=True)
    ap.add_argument("--stop-step", type=int)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    run(a.plan, a.arm, a.stop_step, a.resume)


if __name__ == "__main__":
    main()
