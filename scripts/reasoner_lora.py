"""LoRA fine-tune of the reasoner (jeff.games.reasoner) on the data from scripts/reasoner_data.py, with MLX.

`mlx_lm.lora` runs out of GPU memory on Qwen3.5 on a 32 GB Mac. In training mode each Gated DeltaNet layer runs a
Python loop over the tokens in place of its Metal kernel, and mlx-lm's trainer compiles each step (`mx.compile`), which
unrolls that loop, forward and backward, into one very large graph. This driver runs the steps uncompiled, adapts only
the top `--layers` layers, and keeps each Gated DeltaNet layer below them in inference mode, so those run on the
kernel; no gradient passes through them. The adapter directory loads with `mlx_lm.load(model, adapter_path=...)`.

Usage: python scripts/reasoner_lora.py --data data/reasoner --out checkpoints/reasoner-lora [--iters 1000]"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.trainer import default_loss, evaluate, grad_checkpoint, iterate_batches
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import save_config

LORA = {"rank": 8, "dropout": 0.0, "scale": 20.0}  # mlx_lm.lora's defaults
REPORT_EVERY, EVAL_EVERY, VALID_BATCHES = 25, 200, 25


class FrozenGatedDeltaNet(GatedDeltaNet):
    """A Gated DeltaNet layer that stays in inference mode, so it runs on the Metal kernel."""

    def _set_training_mode(self, mode: bool) -> None:
        self._training = False


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="checkpoints/base-0.8b")
    parser.add_argument("--data", type=Path, required=True, help="Directory with train.jsonl and valid.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Adapter directory")
    parser.add_argument("--layers", type=int, default=4, help="Top layers to adapt")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=1024, help="Longer rows are cut")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    model, tokenizer = load(args.model)
    model.freeze()
    linear_to_lora_layers(model, args.layers, LORA)
    for layer in model.layers[:len(model.layers) - args.layers]:
        if layer.is_linear:
            layer.linear_attn.__class__ = FrozenGatedDeltaNet
    train_set, valid_set = (CacheDataset(ChatDataset(read(args.data / f"{split}.jsonl"), tokenizer, mask_prompt=True))
                            for split in ("train", "valid"))
    args.out.mkdir(parents=True)
    save_config({"model": args.model, "fine_tune_type": "lora", "num_layers": args.layers, "lora_parameters": LORA,
                 "iters": args.iters, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                 "data": str(args.data), "mask_prompt": True, "max_tokens": args.max_tokens}, args.out / "adapter_config.json")
    grad_checkpoint(model.layers[0])
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    step = nn.value_and_grad(model, default_loss)
    model.train()
    losses, tokens, started = [], 0, time.perf_counter()
    batches = iterate_batches(train_set, args.batch_size, args.max_tokens, loop=True, seed=0)
    for iteration, batch in zip(range(1, args.iters + 1), batches):
        (loss, count), grad = step(model, *batch)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        tokens += count.item()
        if iteration % REPORT_EVERY == 0:
            print(f"Iter {iteration}: Train loss {sum(losses) / len(losses):.3f}, {tokens / (time.perf_counter() - started):.0f} "
                  f"tokens/s, {(time.perf_counter() - started) / iteration:.2f} s/iter", flush=True)
            losses = []
        if iteration % EVAL_EVERY == 0 or iteration == args.iters:
            valid = evaluate(model, valid_set, args.batch_size, VALID_BATCHES, args.max_tokens)
            model.train()
            print(f"Iter {iteration}: Val loss {valid:.3f}", flush=True)
            mx.save_safetensors(str(args.out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    print(f"Saved final adapters to {args.out}", flush=True)


if __name__ == "__main__":
    main()
