"""LoRA fine-tuning helpers for Qwen3.5 models in MLX on a Mac (scripts/reasoner_lora.py, scripts/jeff_lora.py).

In training mode each Gated DeltaNet layer runs a Python loop over the tokens in place of its Metal kernel, and the
backward pass keeps each token's state. On a 32 GB Mac that fits only when few layers carry gradients, so `adapt` puts
LoRA on the top layers only and keeps the Gated DeltaNet layers below them in inference mode: they run on the kernel,
and no gradient passes through them. Training steps must not be compiled (`mx.compile`), which unrolls the loop into
one graph that does not fit."""

from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.tuner.utils import linear_to_lora_layers

LORA = {"rank": 8, "dropout": 0.0, "scale": 20.0}  # mlx_lm.lora's defaults


class FrozenGatedDeltaNet(GatedDeltaNet):
    """A Gated DeltaNet layer that stays in inference mode, so it runs on the Metal kernel."""

    def _set_training_mode(self, mode: bool) -> None:
        self._training = False


def adapt(model, layers: int) -> None:
    """Freeze a Qwen3.5 model, add LoRA to its top `layers` layers, and keep the Gated DeltaNet layers below them on
    the kernel."""
    model.freeze()
    linear_to_lora_layers(model, layers, LORA)
    for layer in model.layers[:len(model.layers) - layers]:
        if layer.is_linear:
            layer.linear_attn.__class__ = FrozenGatedDeltaNet
