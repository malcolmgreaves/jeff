"""LoRA fine-tune of the reasoner (jeff.games.reasoner) on the data from scripts/reasoner_data.py, with MLX.

`mlx_lm.lora` runs out of GPU memory on Qwen3.5 on a 32 GB Mac. In training mode each Gated DeltaNet layer runs a
Python loop over the tokens in place of its Metal kernel, and mlx-lm's trainer compiles each step (`mx.compile`), which
unrolls that loop, forward and backward, into one very large graph. This driver runs the steps uncompiled and adapts
only the top `--layers` layers (jeff.mlx_tuning). Every `--eval-every` steps it scores the validation set and keeps the
adapter with the lowest validation loss; with `--patience` it stops once that many validations pass without a lower
loss. With `--reduce-on-plateau N`, running out of patience first halves the learning rate and goes back to the best
adapter, up to N times, before the run stops. `--resume` starts from an earlier adapter's weights. The adapter directory loads with
`mlx_lm.load(model, adapter_path=...)`.

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
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.trainer import default_loss, evaluate, grad_checkpoint, iterate_batches
from mlx_lm.utils import save_config

from jeff.mlx_tuning import LORA, adapt

REPORT_EVERY = 25


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
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--valid-batches", type=int, default=25, help="-1 scores the whole validation set")
    parser.add_argument("--patience", type=int, help="Stop after this many validations without a lower validation loss")
    parser.add_argument("--resume", type=Path, help="Start from this adapter directory's weights, with a fresh optimizer")
    parser.add_argument("--reduce-on-plateau", type=int, default=0,
                        help="Times to halve the learning rate and go back to the best adapter when patience runs out")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    model, tokenizer = load(args.model)
    adapt(model, args.layers)
    if args.resume:
        model.load_weights(str(args.resume / "adapters.safetensors"), strict=False)
    train_set, valid_set = (CacheDataset(ChatDataset(read(args.data / f"{split}.jsonl"), tokenizer, mask_prompt=True))
                            for split in ("train", "valid"))
    args.out.mkdir(parents=True)
    config = {"model": args.model, "fine_tune_type": "lora", "num_layers": args.layers, "lora_parameters": LORA,
              "iters": args.iters, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "data": str(args.data), "mask_prompt": True, "max_tokens": args.max_tokens,
              "resume": str(args.resume) if args.resume else None, "patience": args.patience,
              "reduce_on_plateau": args.reduce_on_plateau}
    save_config(config, args.out / "adapter_config.json")
    grad_checkpoint(model.layers[0])
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    step = nn.value_and_grad(model, default_loss)

    def validate(iteration: int) -> float:
        valid = evaluate(model, valid_set, args.batch_size, args.valid_batches, args.max_tokens)
        model.train()
        print(f"Iter {iteration}: Val loss {valid:.4f}", flush=True)
        return valid

    def save() -> None:
        mx.save_safetensors(str(args.out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))

    best, best_iteration, stale, learning_rate, reductions = validate(0), 0, 0, args.learning_rate, 0
    save()
    model.train()
    losses, tokens, started = [], 0, time.perf_counter()
    batches = iterate_batches(train_set, args.batch_size, args.max_tokens, loop=True, seed=0)
    iteration = 0
    for iteration, batch in zip(range(1, args.iters + 1), batches):
        (loss, count), grad = step(model, *batch)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(loss.item())
        tokens += count.item()
        if iteration % REPORT_EVERY == 0:
            print(f"Iter {iteration}: Train loss {sum(losses) / len(losses):.4f}, {tokens / (time.perf_counter() - started):.0f} "
                  f"tokens/s, {(time.perf_counter() - started) / iteration:.2f} s/iter", flush=True)
            losses = []
        if iteration % args.eval_every == 0 or iteration == args.iters:
            valid = validate(iteration)
            if valid < best:
                best, best_iteration, stale = valid, iteration, 0
                save()
            else:
                stale += 1
            if args.patience is not None and stale >= args.patience:
                if reductions < args.reduce_on_plateau:
                    reductions, stale, learning_rate = reductions + 1, 0, learning_rate / 2
                    model.load_weights(str(args.out / "adapters.safetensors"), strict=False)
                    optimizer = optim.Adam(learning_rate=learning_rate)
                    print(f"Iter {iteration}: learning rate halved to {learning_rate:g}; back to the adapter from iteration "
                          f"{best_iteration}", flush=True)
                    continue
                print(f"Stopped: {stale} validations without a lower loss", flush=True)
                break
    save_config({**config, "iterations_run": iteration, "best_iteration": best_iteration, "best_valid_loss": best,
                 "learning_rate_reductions": reductions},
                args.out / "adapter_config.json")
    print(f"Saved final adapters to {args.out} (iteration {best_iteration}, validation loss {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
