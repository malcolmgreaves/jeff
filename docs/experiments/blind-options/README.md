# Options that don't state their outcome

This experiment measures how well Jeff chooses between moves when the options don't say what each move leads to. It
uses the Frogger and Pac-Man harnesses in `jeff.games`. The README's game results use the `outcomes` wording, where each
option states the consequence of the move and whether it's good ("you would be hit by a car and lose a life"). This
experiment removes that information from the options in steps and scores every choice against the game's own facts.

Jeff-0.8B picks well only when each option states what happens and whether that's good. When the options state the
facts without a verdict, its choices depend on the facts much less. When the facts are only in the state, its choices
don't depend on them at all. An untrained Qwen3.5-0.8B control shows that Jeff's fine-tuning improves the first case,
doesn't fix the move-only case, and makes one wording worse.

## Method

`scripts/blind_options.py` drives each game with the rule bot, with 25% of its moves random so that risky states occur.
At every state, the script asks the server about the same state in six option wordings:

- **`outcomes`**: each option states what the move leads to and whether that's good or bad. This is the README's wording.
- **`situation`**: each option states what is where the move leads, without a verdict ("a car will be on that square").
- **`listed`**: each option is only its key ("up", "keep going"). The state lists the `situation` sentences under the
  same keys.
- **`judgement`**: each option only names the move ("Move one square up"). The facts are fields of the state, one for each
  direction.
- **`board`**: the same as `judgement`, with those fields removed, so only the board is left.
- **`rule`**: each option states the condition that makes it the right move, based on the state's fields.

The driver doesn't depend on the model, so every wording and every model sees exactly the same states. Each game
contributes 400 states from episodes with seeds 1234 and later, at most 60 turns each.

The script reports these measures:

- **Safe choice**: the share of contested states where the chosen move doesn't lose a life this turn. A state is
  contested when some legal moves are safe and some aren't. In Frogger, a move is safe when its square is free after
  the cars and logs move. In Pac-Man, a move is safe when no ghost is within 2 steps that way. A ghost that close meets
  Pac-Man this turn when it moves toward him, and every wording can express that distance.
- **Fact sensitivity**: P(picked | option safe) minus P(picked | option unsafe), averaged over option keys. A player that
  always picks the same key scores 0. The measure has noise of about ±0.05. The rule bot's value is the reference for a
  player that uses the facts fully.
- **Agrees with rule bot**: the share of all states where the choice matches the rule bot, which also rewards progress.
- **Baselines**: random moves, the safest fixed option (always choosing one of the game's named moves), and the rule bot.
  The rule bot always picks a safe move when one exists.

The control model is Qwen3.5-0.8B with its readout copied from its language-model head and no fine-tuning. That is
how `jeff.model.DecisionModel` starts before training.

## Results

The runs used the MLX backend on an Apple M1 Max. The files `jeff-0.8b.json` and `untrained-0.8b.json` in this
directory hold every state, choice, and probability.

### Frogger (289 contested states)

| Wording | Safe: Jeff | Safe: untrained | Sensitivity: Jeff | Sensitivity: untrained |
|---|---|---|---|---|
| Random moves | 67.5% | 67.5% | 0.00 | 0.00 |
| Always "stay" | 80.6% | 80.6% | 0.00 | 0.00 |
| Rule bot | 100% | 100% | +0.39 | +0.39 |
| `outcomes` | 100.0% | 92.4% | +0.38 | +0.29 |
| `situation` | 87.9% | 84.1% | +0.14 | +0.10 |
| `listed` | 80.6% | 69.6% | 0.00 | +0.06 |
| `judgement` | 59.9% | 30.8% | −0.05 | 0.00 |
| `board` | 58.8% | 30.8% | −0.02 | 0.00 |
| `rule` | 51.6% | 73.0% | −0.03 | −0.06 |

### Pac-Man (188 contested states)

| Wording | Safe: Jeff | Safe: untrained | Sensitivity: Jeff | Sensitivity: untrained |
|---|---|---|---|---|
| Random moves | 56.6% | 56.6% | 0.00 | 0.00 |
| Always "keep going" (legal in 136) | 89.7% | 89.7% | 0.00 | 0.00 |
| Rule bot | 100% | 100% | +0.75 | +0.75 |
| `outcomes` | 97.9% | 79.3% | +0.65 | +0.38 |
| `situation` | 64.9% | 56.4% | +0.33 | +0.17 |
| `listed` | 16.0% | 86.2% | −0.60 | −0.05 |
| `judgement` | 67.0% | 67.0% | +0.02 | +0.11 |
| `board` | 45.7% | 67.0% | −0.08 | +0.11 |
| `rule` | 73.9% | 68.1% | +0.19 | +0.13 |

### Jeff minus untrained, paired on the same contested states

| Wording | Frogger safe (95% CI) | Pac-Man safe (95% CI) |
|---|---|---|
| `outcomes` | +7.6 (+4.5 to +10.7) | +18.6 (+12.7 to +24.6) |
| `situation` | +3.8 (−1.8 to +9.4) | +8.5 (−1.8 to +18.8) |
| `listed` | +11.1 (+5.3 to +16.9) | −70.2 (−77.2 to −63.2) |
| `judgement` | +29.1 (+21.7 to +36.4) | 0.0 (−8.0 to +8.0) |
| `board` | +28.0 (+21.2 to +34.8) | −21.3 (−33.4 to −9.2) |
| `rule` | −21.5 (−29.8 to −13.1) | +5.9 (−2.3 to +14.0) |

## Findings

- **With outcomes in the options, fine-tuning helps.** Jeff is safer than the untrained model in both games, and its
  fact sensitivity is close to the rule bot's.
- **Without the good-or-bad verdict, performance drops.** With `situation`, Jeff's fact sensitivity falls by more than
  half in both games. In Frogger it waits in 72% of contested states: when hopping forward is safe, it hops only 29% of
  the time, compared with 96% under `outcomes`. Its gain over the untrained model isn't statistically clear.
- **With move-only options, neither model uses the facts.** Fact sensitivity for `judgement` and `board` is between
  −0.08 and +0.11 for both models. Jeff's 29-point Frogger gain comes from a different fixed habit, not from reading
  the state. The untrained model hops forward in 100% of contested states, and Jeff waits in 60-67% of them. In the
  `judgement` wording, Jeff picks "up" with probability 0.26 when the square above is safe and 0.29 when a car will be
  there.
- **Fine-tuning caused the reversal in `listed`.** In Pac-Man, the untrained model mostly ignores the state sentences
  (−0.05). Jeff moves toward the danger they describe (−0.60): it picks "reverse" in 86% of the states where the state
  says "a ghost is right there" for that move, and in 14% of those where reversing is safe.

For decisions like these, compute each option's consequence in code and write it into the option, including whether
it's good ("you lose a life"). This matches the README's advice to "reason in code, decide with Jeff."

## Limits

- **Coverage:** the experiment tests one phrasing for each wording, only the 0.8B model, and one state distribution.
- **Pac-Man safety:** the 2-step definition follows the `outcomes` wording's "deadly" boundary. The rule bot uses 3
  steps.
- **Control temperature:** the untrained checkpoint uses temperature 1. That changes only the "probability on safe
  moves" column in the JSON files, not the choices or the measures in these tables.
- **Scope:** the experiment tests choices between actions whose consequences decide the outcome. It doesn't measure
  ordinary text classification, where the evidence is in the text and the options are labels.

## Reproduce the results

1. Install the MLX extra and download the Jeff checkpoint:

   ```bash
   uv sync --extra mac
   uv run hf download mstrasser/Jeff-Qwen3.5-0.8B --local-dir checkpoints/jeff-0.8b
   ```

1. Build the untrained control checkpoint:

   ```python
   import torch
   from jeff.model import DecisionModel

   model = DecisionModel(base_model="Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17",
                         device="cpu", cache_dir=".cache/huggingface/hub")
   model.backbone.to(torch.bfloat16)
   model.readout.to(torch.bfloat16)
   model.save("checkpoints/untrained-0.8b", temperature=1.0)
   ```

1. Start a server for one checkpoint, run the experiment against it, then repeat with the other checkpoint:

   ```bash
   JEFF_BACKEND=mlx JEFF_CHECKPOINT=checkpoints/jeff-0.8b PORT=8765 uv run jeff-serve
   uv run python scripts/blind_options.py http://127.0.0.1:8765 --out runs/blind/jeff-0.8b.json --states 400
   ```

MLX inference was deterministic in these runs: a repeated Jeff run reproduced every choice.
