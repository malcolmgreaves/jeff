"""LoRA fine-tune of a Jeff checkpoint with MLX, for a Mac (the repository's trainer, jeff.train, needs CUDA).

The top `--layers` layers get LoRA weights; with `--train-readout`, Jeff's answer readout is trained too. Everything
else stays frozen (jeff.mlx_tuning). The loss is cross-entropy over each row's options against its label, as in jeff.train, with the
options shuffled at every step. Gradients are summed over `--accumulate` rows per update. Every `--eval-every` updates
the development fold is scored, and the update with the lowest development loss is kept. The temperature is then
fitted on the calibration fold (jeff.evaluate.fit_temperature). The adapter directory loads with
MlxDecisionModel(checkpoint, adapter), or JEFF_ADAPTER with the MLX server backend.

Usage: python scripts/jeff_lora.py --checkpoint checkpoints/jeff-0.8b --data DIR --out ADAPTER_DIR
       (DIR holds train.jsonl, dev.jsonl and calibration.jsonl rows, as jeff.types.Example)"""

import argparse
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from jeff.evaluate import fit_temperature, read_rows
from jeff.mlx_backend import MlxDecisionModel
from jeff.mlx_tuning import LORA, adapt
from jeff.types import Example


class Decider(nn.Module):
    """Jeff in MLX: the backbone's last hidden state times the readout gives one logit per answer code."""

    def __init__(self, backbone: nn.Module, readout: mx.array) -> None:
        super().__init__()
        self.backbone = backbone
        self.readout = readout

    def __call__(self, ids: mx.array) -> mx.array:
        hidden = self.backbone.language_model.model(ids)[0, -1].astype(mx.float32)
        return self.readout @ hidden


def loss(decider: Decider, ids: mx.array, count: int, label: int) -> mx.array:
    return nn.losses.cross_entropy(decider(ids)[None, :count], mx.array([label]), reduction="mean")


def shuffled(row: Example, rng: random.Random) -> Example:
    """The row with its options in a random order (the label is a key, so it follows)."""
    result = deepcopy(row)
    items = list(result["question"]["criteria"].items())  # type: ignore[typeddict-item]
    rng.shuffle(items)
    result["question"]["criteria"] = dict(items)  # type: ignore[typeddict-item]
    return result


def encode(jeff: MlxDecisionModel, row: Example) -> tuple[mx.array, int, int]:
    keys = list(row["question"]["criteria"])  # type: ignore[typeddict-item]
    return mx.array([jeff.prompt_ids(row)]), len(keys), keys.index(str(row["label"]))


def score(decider: Decider, jeff: MlxDecisionModel, rows: list[Example]) -> tuple[float, float, list[list[float]], list[int]]:
    """Mean loss and accuracy on fixed rows (options in their saved order), with each row's logits and label index."""
    decider.eval()
    logits, labels, losses = [], [], []
    for row in rows:
        ids, count, label = encode(jeff, row)
        values = decider(ids)[:count]
        losses.append(nn.losses.cross_entropy(values[None], mx.array([label])).item())
        logits.append(values.tolist())
        labels.append(label)
    decider.train()
    accuracy = sum(max(range(len(v)), key=v.__getitem__) == label for v, label in zip(logits, labels)) / len(rows)
    return sum(losses) / len(losses), accuracy, logits, labels


def save(decider: Decider, out: Path) -> None:
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(decider.backbone.trainable_parameters())))
    mx.save_safetensors(str(out / "readout.safetensors"), {"weight": decider.readout})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4, help="Top layers to adapt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8, help="Rows per update")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=25, help="Updates between development scores")
    parser.add_argument("--train-readout", action="store_true",
                        help="Also train the answer readout; it was trained with a far lower learning rate than LoRA's")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    train, dev, calibration = (read_rows(args.data / f"{name}.jsonl") for name in ("train", "dev", "calibration"))
    jeff = MlxDecisionModel(args.checkpoint)
    decider = Decider(jeff.model, jeff.readout)
    decider.freeze()
    adapt(decider.backbone, args.layers)
    if args.train_readout:
        decider.unfreeze(keys=["readout"], recurse=False)
    decider.train()
    args.out.mkdir(parents=True)
    step = nn.value_and_grad(decider, loss)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    rng = random.Random(args.seed)
    best_loss, best_accuracy = score(decider, jeff, dev)[:2]
    best_update, updates, summed, pending, losses = 0, 0, None, 0, []
    print(json.dumps({"update": 0, "dev_loss": best_loss, "dev_accuracy": best_accuracy}), flush=True)
    save(decider, args.out)
    started = time.perf_counter()
    for _ in range(args.epochs):
        for index in rng.sample(range(len(train)), len(train)):
            ids, count, label = encode(jeff, shuffled(train[index], rng))
            value, grads = step(decider, ids, count, label)
            summed = grads if summed is None else tree_map(mx.add, summed, grads)
            mx.eval(value, summed)
            losses.append(value.item())
            pending += 1
            if pending < args.accumulate:
                continue
            optimizer.update(decider, tree_map(lambda g: g / pending, summed))
            mx.eval(decider.parameters(), optimizer.state)
            summed, pending, updates = None, 0, updates + 1
            if updates % args.eval_every == 0:
                dev_loss, dev_accuracy = score(decider, jeff, dev)[:2]
                improved = dev_loss < best_loss
                if improved:
                    best_loss, best_accuracy, best_update = dev_loss, dev_accuracy, updates
                    save(decider, args.out)
                print(json.dumps({"update": updates, "train_loss": sum(losses) / len(losses), "dev_loss": dev_loss,
                                  "dev_accuracy": dev_accuracy, "saved": improved,
                                  "seconds": round(time.perf_counter() - started)}), flush=True)
                losses = []
    decider.backbone.load_weights(str(args.out / "adapters.safetensors"), strict=False)  # the kept update
    decider.readout = mx.load(str(args.out / "readout.safetensors"))["weight"]
    _, _, logits, labels = score(decider, jeff, calibration)
    temperature = fit_temperature(logits, labels)
    if not math.isfinite(temperature):
        raise ValueError("The fitted temperature is not finite")
    config = {"checkpoint": args.checkpoint, "fine_tune_type": "lora", "num_layers": args.layers, "lora_parameters": LORA,
              "temperature": temperature, "data": str(args.data), "epochs": args.epochs, "accumulate": args.accumulate,
              "learning_rate": args.learning_rate, "train_readout": args.train_readout, "seed": args.seed,
              "updates": updates, "kept_update": best_update, "dev_loss": best_loss, "dev_accuracy": best_accuracy, "train_rows": len(train), "dev_rows": len(dev),
              "calibration_rows": len(calibration)}
    (args.out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)


if __name__ == "__main__":
    main()
