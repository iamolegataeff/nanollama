"""
LoRA SFT for nanollama.

Replaces full-finetune personality extraction with LoRA adapters.
Inspired by Yent (github.com/ariannamethod/yent): rank 64, per-voice adapters.

Usage:
    # LoRA fine-tune on text data
    python -m scripts.chat_sft --base-checkpoint path/to/checkpoint.pt \\
        --data path/to/data.txt --voice myvoice --rank 64

    # LoRA fine-tune on JSONL conversations
    python -m scripts.chat_sft --base-checkpoint path/to/checkpoint.pt \\
        --data path/to/conversations.jsonl --voice myvoice

    # With Chuck optimizer
    python -m scripts.chat_sft --base-checkpoint path/to/checkpoint.pt \\
        --data path/to/data.txt --voice myvoice --optimizer chuck

Data formats:
    .txt  — raw text, trained as next-token prediction
    .jsonl — one JSON object per line with "text" field, or
             {"messages": [{"role": "user", "content": "..."}, ...]}

Output:
    ~/.cache/nanollama/lora/<voice>/adapter.pt    — LoRA weights only
    ~/.cache/nanollama/lora/<voice>/merged.pt      — full merged checkpoint
"""

import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from nanollama.common import (
    compute_init, compute_cleanup, print0,
    autodetect_device_type, get_base_dir,
)
from nanollama.llama import Llama, LlamaConfig
from nanollama.lora import apply_lora, merge_lora, save_lora, lora_params


def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT for nanollama")

    # Model
    p.add_argument("--base-checkpoint", type=str, required=True,
                   help="Path to base model checkpoint (.pt)")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Override vocab size (auto-detected from checkpoint)")

    # LoRA
    p.add_argument("--rank", type=int, default=64, help="LoRA rank (default: 64)")
    p.add_argument("--alpha", type=float, default=64.0, help="LoRA alpha (default: 64)")
    p.add_argument("--targets", type=str, default=None,
                   help="Comma-separated target modules (default: all attention + FFN)")

    # Data
    p.add_argument("--data", type=str, required=True,
                   help="Training data: .txt (raw text) or .jsonl")
    p.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")

    # Training
    p.add_argument("--voice", type=str, default="default", help="Voice/adapter name")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "chuck"], help="Optimizer")
    p.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    p.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N steps (0=only final)")

    # Output
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: ~/.cache/nanollama/lora/<voice>)")

    return p.parse_args()


def load_text_data(path: str, tokenizer, max_seq_len: int):
    """Load .txt file and chunk into sequences."""
    with open(path) as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    # Chunk into max_seq_len + 1 sequences (input + target)
    chunks = []
    for i in range(0, len(tokens) - max_seq_len, max_seq_len):
        chunks.append(tokens[i:i + max_seq_len + 1])
    if len(tokens) > max_seq_len + 1:
        # Last chunk
        chunks.append(tokens[-(max_seq_len + 1):])
    print0(f"Loaded {len(text):,} chars → {len(tokens):,} tokens → {len(chunks)} chunks")
    return chunks


def load_jsonl_data(path: str, tokenizer, max_seq_len: int):
    """Load .jsonl file. Supports {"text": "..."} or {"messages": [...]}."""
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "text" in obj:
                text = obj["text"]
            elif "messages" in obj:
                # Render as: User: ...\nAssistant: ...\n
                parts = []
                for msg in obj["messages"]:
                    role = msg.get("role", "user").capitalize()
                    parts.append(f"{role}: {msg['content']}")
                text = "\n".join(parts)
            else:
                continue

            tokens = tokenizer.encode(text)
            if len(tokens) > max_seq_len + 1:
                tokens = tokens[:max_seq_len + 1]
            if len(tokens) < 10:
                continue
            chunks.append(tokens)

    print0(f"Loaded {len(chunks)} conversations from JSONL")
    return chunks


def collate_batch(chunks, batch_indices, max_seq_len, device):
    """Pad and collate a batch of token sequences."""
    batch = []
    for idx in batch_indices:
        tokens = chunks[idx]
        # Pad to max_seq_len + 1
        if len(tokens) < max_seq_len + 1:
            pad_id = 0
            tokens = tokens + [pad_id] * (max_seq_len + 1 - len(tokens))
        batch.append(tokens[:max_seq_len + 1])

    t = torch.tensor(batch, dtype=torch.long, device=device)
    return t[:, :-1], t[:, 1:]  # inputs, targets


def main():
    args = parse_args()

    # Compute init
    device_type = autodetect_device_type()
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    autocast_ctx = (torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
                    if device_type == "cuda" else nullcontext())

    # Load base model
    print0(f"Loading base model from {args.base_checkpoint}")
    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", {})
    config = LlamaConfig(**config_dict)
    if args.vocab_size:
        config.vocab_size = args.vocab_size

    model = Llama(config)
    state = checkpoint["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.train()

    base_step = checkpoint.get("step", 0)
    print0(f"Base model: {sum(p.numel() for p in model.parameters()):,} params, trained to step {base_step}")

    # Apply LoRA
    targets = args.targets.split(",") if args.targets else None
    apply_lora(model, rank=args.rank, alpha=args.alpha, target_modules=targets)

    # Load tokenizer
    from nanollama.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()

    # Load data
    print0(f"Loading data from {args.data}")
    if args.data.endswith(".jsonl"):
        chunks = load_jsonl_data(args.data, tokenizer, args.max_seq_len)
    else:
        chunks = load_text_data(args.data, tokenizer, args.max_seq_len)

    if not chunks:
        print0("ERROR: No training data loaded!")
        return

    # Setup optimizer
    print0("Setting up optimizer...")
    if args.optimizer == "chuck":
        from nanollama.chuck import ChuckOptimizer
        params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": params, "lr": args.lr}]
        optimizer = ChuckOptimizer(param_groups, lr=args.lr)
        print0(f"Chuck optimizer: {len(params)} trainable params")
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        print0(f"AdamW optimizer: {len(params)} trainable params")

    # Output directory
    output_dir = args.output_dir or os.path.join(get_base_dir(), "lora", args.voice)
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    total_steps = (len(chunks) // args.batch_size) * args.epochs
    print0(f"\nStarting LoRA SFT: {args.epochs} epochs, {len(chunks)} samples, "
           f"batch_size={args.batch_size}, total_steps={total_steps}")
    print0(f"Voice: {args.voice}, rank={args.rank}, alpha={args.alpha}")
    print0(f"Output: {output_dir}\n")

    step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        # Shuffle
        indices = torch.randperm(len(chunks)).tolist()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_start in range(0, len(indices), args.batch_size):
            batch_idx = indices[batch_start:batch_start + args.batch_size]
            if len(batch_idx) < args.batch_size:
                continue

            inputs, targets = collate_batch(chunks, batch_idx, args.max_seq_len, device)

            with autocast_ctx:
                loss = model(inputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            step += 1

            if step % args.log_every == 0:
                avg = epoch_loss / epoch_steps
                print0(f"epoch {epoch+1}/{args.epochs} | step {step}/{total_steps} | "
                       f"loss {loss_val:.4f} | avg {avg:.4f}")

            if args.save_every > 0 and step % args.save_every == 0:
                save_lora(model, os.path.join(output_dir, f"adapter_step{step}.pt"))

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print0(f"Epoch {epoch+1} done — avg loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_lora(model, os.path.join(output_dir, "adapter_best.pt"))

    # Save final adapter
    save_lora(model, os.path.join(output_dir, "adapter.pt"))

    # Save merged model
    print0("Merging LoRA into base weights...")
    merge_lora(model)
    merged_path = os.path.join(output_dir, "merged.pt")
    merged_state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": merged_state,
        "step": base_step,
        "config": config_dict,
        "lora": {"rank": args.rank, "alpha": args.alpha, "voice": args.voice},
    }, merged_path)
    print0(f"Merged model saved to {merged_path}")

    # Save metadata
    meta = {
        "voice": args.voice,
        "rank": args.rank,
        "alpha": args.alpha,
        "base_checkpoint": args.base_checkpoint,
        "data": args.data,
        "epochs": args.epochs,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "best_loss": best_loss,
        "total_steps": step,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print0(f"\nLoRA SFT complete! Voice: {args.voice}, best loss: {best_loss:.4f}")
    print0(f"  Adapter: {output_dir}/adapter.pt")
    print0(f"  Merged:  {merged_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
