# Options that don't state their outcome

This experiment measures how well Jeff chooses between moves when the options don't say what each move leads to. It
uses the Frogger and Pac-Man harnesses in `jeff.games`. The README's game results use the `outcomes` wording, where each
option states the consequence of the move and whether it's good ("you would be hit by a car and lose a life"). This
experiment removes that information from the options in steps and scores every choice against the game's own facts.

Jeff-0.8B picks well only when each option states what happens and whether that's good. When the options state the
facts without a verdict, its choices depend on the facts much less. When the facts are only in the state, its choices
don't depend on them at all. An untrained Qwen3.5-0.8B control shows that Jeff's fine-tuning improves the first case,
doesn't fix the move-only case, and makes one wording worse.

A second experiment puts a generative model in front of Jeff. The model reads the full game state and writes what each
move leads to, and Jeff chooses from those outcomes. With the untrained Qwen3.5-0.8B as that reasoner, the pair does no
better than random. After a LoRA fine-tune of the reasoner, the pair matches code-written outcomes in Pac-Man and
reaches 79% safe choices in Frogger. There, every miss traces back to the reasoner's mistakes. For the details, see
[Two-stage decisions](#two-stage-decisions).

A third experiment gives the fine-tuned reasoner only the board, so it has to work the outcomes out itself. It misses
many hazards, and the pair reaches 73% safe choices in Frogger and 62% in Pac-Man. For the details, see
[Board-only reasoner](#board-only-reasoner).

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

## Two-stage decisions

This experiment replaces the code that writes each option's outcome with a generative model, the reasoner.

### Method

For each state, the reasoner (Qwen3.5-0.8B, generating text) gets the full state that the `judgement` wording uses: the
board plus the facts for each direction. It also gets the legal moves. Its prompt (`jeff.games.reasoner.messages`)
gives the game's rules, explains the fields for each direction, and asks for one line for each move in the form
`<move>: <what the state shows> => <what happens>`. The code starts the reply with the first move's name. Decoding is
greedy, with no thinking block. Jeff then chooses from options that pair each move's name with the outcome the reasoner
wrote, along with the short state the `outcomes` wording uses. The `reasoned` condition in `scripts/blind_options.py`
runs this pipeline on the same 400 states in each game as the other wordings.

The experiment tests two reasoners:

- **Untrained**: the released Qwen3.5-0.8B. I tried four prompt variants on development seed 5000, which isn't an
  evaluation seed, and none beat chance on those states. With an example line that used a real move name, the model
  copied the example onto that move. With no example, it wrote coordinates in place of outcomes. The final prompt uses
  an example with a made-up move ("jump").
- **Fine-tuned**: a LoRA adapter trained on the same prompts. The answers come from the game's own facts, in the
  `outcomes` wording's phrases (`jeff.games.reasoner.target`). `scripts/reasoner_data.py` writes 4,000 training states
  from seeds 100,000 and later and 200 validation states from seeds 200,000 and later, half from each game.
  `scripts/reasoner_lora.py` trains rank-8 LoRA on the top 4 of 24 layers, for 1,000 steps of 2 rows at learning rate
  1e-4. Training took about 60 minutes on the M1 Max, and validation loss fell from 1.43 to 0.008.

The standard `mlx_lm.lora` command runs out of GPU memory on this model on a 32 GB Mac. In training mode, each Gated
DeltaNet layer runs a Python loop over the tokens in place of its Metal kernel, and the trainer compiles each step. So
`scripts/reasoner_lora.py` runs uncompiled steps, and keeps the Gated DeltaNet layers that come before the adapted ones
on the kernel.

### Results

Jeff-0.8B chose in every row of these tables. The first rows repeat earlier results for comparison.

| Frogger (289 contested states) | Safe choice | Fact sensitivity | Agrees with rule bot |
|---|---|---|---|
| Random moves | 67.5% | 0.00 | 21.4% |
| Always "stay" | 80.6% | 0.00 | |
| `outcomes` (code writes the outcomes) | 100.0% | +0.38 | 96.0% |
| `situation` | 87.9% | +0.14 | 53.2% |
| `judgement` | 59.9% | −0.05 | 46.2% |
| Untrained reasoner, then Jeff | 66.8% | −0.01 | 43.0% |
| Fine-tuned reasoner, then Jeff | 79.2% | +0.21 | 81.5% |

| Pac-Man (188 contested states) | Safe choice | Fact sensitivity | Agrees with rule bot |
|---|---|---|---|
| Random moves | 56.6% | 0.00 | 40.7% |
| Always "keep going" (legal in 136) | 89.7% | 0.00 | |
| `outcomes` (code writes the outcomes) | 97.9% | +0.65 | 77.5% |
| `situation` | 64.9% | +0.33 | 45.2% |
| `judgement` | 67.0% | +0.02 | 46.8% |
| Untrained reasoner, then Jeff | 46.3% | −0.06 | 34.2% |
| Fine-tuned reasoner, then Jeff | 98.9% | +0.69 | 65.0% |

Compared with `outcomes` on the same contested states, the untrained pair is 28 to 39 points less safe in Frogger and
44 to 59 points less safe in Pac-Man (95% CIs). The fine-tuned pair is 16 to 25 points less safe in Frogger, and in
Pac-Man the difference is between −1.5 and +3.6 points.

### What the reasoners wrote

- **Untrained, Frogger:** its verdicts are close to random. It says the frog loses a life for 56% of unsafe moves and
  for 44% of safe moves, going by a keyword check of its text.
- **Untrained, Pac-Man:** it copied the prompt's example outcome ("no ghost can reach Pac-Man soon, and he eats a
  pellet right away") for 907 of the 941 moves it answered. It answered 92% of the legal moves. Jeff got nearly the same
  text for every option.
- **Fine-tuned, Pac-Man:** it wrote the exact correct outcome for all 1,024 moves.
- **Fine-tuned, Frogger:** it wrote the exact correct outcome for 89.2% of moves, and the correct lose-a-life verdict for
  93.7%. Its errors fall in the contested states. In the 171 contested states where every outcome was exact, Jeff chose
  a safe move 100% of the time. In the 118 with at least one error, Jeff chose a safe move 49.2% of the time. The most
  harmful error, in 34 moves, wrote "safe, and one row closer to the goal" where a car will be. The most common error,
  in 55 moves, wrote "fall in the water" in place of "hit by a car," which is wrong but still says a life is lost.

### Findings

- **Jeff chooses correctly from correct outcomes, whoever writes them.** When the reasoner's outcomes were all
  correct, Jeff's choices were as safe as with the code-written outcomes in both games.
- **The pipeline is only as good as the reasoner.** The untrained 0.8B reasoner can't turn the state's facts into
  outcomes, so the pair does no better than random. The fine-tuned reasoner learns the Pac-Man mapping fully and the
  Frogger mapping mostly, and the pair's safety follows its accuracy.
- **Generation costs time.** The reasoner adds about 1.5 seconds for each decision on the M1 Max, against about 0.1
  seconds for Jeff alone.

## Board-only reasoner

The fine-tuned reasoner in the two-stage experiment reads the facts for each direction from its input, so its task is
a lookup. This experiment removes those facts: the reasoner sees only the board and must work each outcome out itself.

### Method

- **Input:** `jeff.games.reasoner.board_only` keeps the `judgement` state's board, legend, and the player's position,
  and drops the facts for each direction. The Frogger board as drawn says that a row moves "every 2 turns" but not
  whether it moves this turn, so the outcome can't be worked out from it. Each road and river row therefore also says
  whether it moves at the end of this turn (`Frogger.board(timing=True)`). A player watching the game sees that timing.
- **Output:** the reasoner first copies the squares it needs, numbered by column. In Frogger that's the rows the frog
  can reach, and in Pac-Man the squares within two rows and two columns of Pac-Man. Then it writes one line for each
  move: the square the move leads to, the rule that applies, then "=>" and the outcome in the `outcomes` wording's
  phrases. For example: "up: row 1, column 4; row 1 moves right, so column 4 gets what is now in column 3: C => you
  would be hit by a car and lose a life." In Pac-Man, the rule is that a ghost on the next square, or on a square next
  to it, catches Pac-Man this turn.
- **Training:** the game code writes the target answers (`jeff.games.reasoner.board_target`), and each target checks its
  own working against the game's answer. `scripts/reasoner_data.py --input board` writes 4,000 training states and 200
  validation states from the same seeds as the two-stage experiment's data. The LoRA setup is the same: rank 8 on the
  top 4 layers, learning rate 1e-4. It trained for 1,500 steps of 2 rows with a 1,200-token limit, which took about 2.1
  hours. Validation loss fell from 0.085 at step 200 to 0.030 at step 1,400 and 0.033 at step 1,500. The script saved
  the adapter from step 1,500.
- **Evaluation:** the `reasoned-board` condition in `scripts/blind_options.py` runs the reasoner and Jeff on the same
  400 states in each game. Jeff sees the same options as in the two-stage experiment.

### Results

| Frogger (289 contested states) | Safe choice | Fact sensitivity | Agrees with rule bot |
|---|---|---|---|
| Random moves | 67.5% | 0.00 | 21.4% |
| Always "stay" | 80.6% | 0.00 | |
| `outcomes` (code writes the outcomes) | 100.0% | +0.38 | 96.0% |
| `board` (Jeff alone, board only) | 58.8% | −0.02 | 53.0% |
| Fine-tuned reasoner with the facts, then Jeff | 79.2% | +0.21 | 81.5% |
| Board-only reasoner, then Jeff | 72.7% | +0.19 | 71.5% |

| Pac-Man (188 contested states) | Safe choice | Fact sensitivity | Agrees with rule bot |
|---|---|---|---|
| Random moves | 56.6% | 0.00 | 40.7% |
| Always "keep going" (legal in 136) | 89.7% | 0.00 | |
| `outcomes` (code writes the outcomes) | 97.9% | +0.65 | 77.5% |
| `board` (Jeff alone, board only) | 45.7% | −0.08 | 34.5% |
| Fine-tuned reasoner with the facts, then Jeff | 98.9% | +0.69 | 65.0% |
| Board-only reasoner, then Jeff | 61.7% | +0.16 | 40.5% |

Compared with `outcomes` on the same contested states, the board-only pair is 22 to 32 points less safe in Frogger and
29 to 43 points less safe in Pac-Man (95% CIs).

### What the board-only reasoner wrote

On the evaluation states:

| Game | Copy of the squares exact | Working exact | Outcome exact | Lose-a-life verdict right |
|---|---|---|---|---|
| Frogger | 5% of states | 83% of moves | 86% of moves | 86% of moves |
| Pac-Man | 21% of states | 19% of moves | 52% of moves | 85% of moves |

- **It misses hazards far more than it invents them.** In Frogger it calls 39% of unsafe moves safe, and 6% of safe
  moves unsafe. In Pac-Man it calls 67% of unsafe moves safe, and 2% of safe moves unsafe. The most common errors write
  a safe outcome where a car will be, or where a ghost is within 2 steps.
- **Its copies are rarely exact.** One wrong character spoils a copy, so this counts any slip. Pac-Man's outcome text
  is often inexact for a second reason: the step counts to the nearest ghost and pellet beyond the copied window are
  often wrong.
- **Jeff follows correct verdicts.** In the contested states where every verdict was right (125 in Frogger, 60 in
  Pac-Man), Jeff chose a safe move 100% of the time. In the states with a wrong verdict (164 and 128), it chose a safe
  move 52% and 44% of the time.

### Findings

- **Working the outcomes out from the board is much harder for the 0.8B reasoner than reading them.** With the same
  training setup, its lose-a-life verdicts are right for 85-86% of moves, against 94% (Frogger) and 100% (Pac-Man) when
  the facts are in its input. Its main error is missing a hazard.
- **The board-only reasoner still helps Jeff.** Jeff alone on the board ignores the facts (sensitivity −0.08 to −0.02).
  With the board-only reasoner in front of it, Jeff's choices depend on them (+0.16 to +0.19). But the pair is less
  safe than the safest fixed option in both games.
- **Jeff's accuracy is set by the reasoner's.** As in the two-stage experiment, Jeff chooses correctly whenever the
  verdicts it reads are right.

## Limits

- **Coverage:** the experiment tests one phrasing for each wording, only the 0.8B model, and one state distribution.
- **Pac-Man safety:** the 2-step definition follows the `outcomes` wording's "deadly" boundary. The rule bot uses 3
  steps.
- **Control temperature:** the untrained checkpoint uses temperature 1. That changes only the "probability on safe
  moves" column in the JSON files, not the choices or the measures in these tables.
- **Scope:** the experiment tests choices between actions whose consequences decide the outcome. It doesn't measure
  ordinary text classification, where the evidence is in the text and the options are labels.
- **What the fine-tuned reasoners learned:** their training answers use the `outcomes` wording's phrases, so a reasoner
  without errors reproduces the `outcomes` condition. With the facts in its input, the reasoner's task is a lookup in
  Frogger and a distance threshold in Pac-Man. The board-only reasoner has to work the outcomes out, but it learned a
  fixed procedure written by the game code.
- **Frogger's `board` row:** the Jeff-alone `board` wording shows the board as drawn, without the lane timing. For rows
  that move every 2 turns, the outcome can't be worked out from that state.
- **Board-only training length:** validation loss was still falling slowly at step 1,500, and the kept adapter wasn't
  the one with the lowest validation loss.
- **Overlap in Pac-Man:** Pac-Man has one maze, so evaluation states can repeat positions seen in training, with
  different ghosts and pellets. Frogger's lanes differ for each seed.
- **Reasoner capacity:** memory limits kept the LoRA adapter to the top 4 layers. More layers or more training might
  improve the Frogger reasoner.

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

1. For the two-stage runs, link the base model that the control step downloaded, and run the untrained reasoner
   against the Jeff server:

   ```bash
   ln -s "$PWD/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17" \
     checkpoints/base-0.8b
   uv run python scripts/blind_options.py http://127.0.0.1:8765 --out runs/blind/reasoned-untrained.json --states 400 \
     --conditions reasoned --reasoner checkpoints/base-0.8b
   ```

1. Build the fine-tuning data and train the adapter. Stop the Jeff server first, to leave the GPU memory for training:

   ```bash
   uv run python scripts/reasoner_data.py --out data/reasoner
   uv run python scripts/reasoner_lora.py --data data/reasoner --out checkpoints/reasoner-lora
   ```

1. Restart the Jeff server and run the fine-tuned reasoner:

   ```bash
   uv run python scripts/blind_options.py http://127.0.0.1:8765 --out runs/blind/reasoned-finetuned.json --states 400 \
     --conditions reasoned --reasoner checkpoints/base-0.8b --adapter checkpoints/reasoner-lora
   ```

1. For the board-only reasoner, build its data, train it, and run it against the Jeff server:

   ```bash
   uv run python scripts/reasoner_data.py --out data/reasoner-board --input board
   uv run python scripts/reasoner_lora.py --data data/reasoner-board --out checkpoints/reasoner-board-lora \
     --iters 1500 --max-tokens 1200
   uv run python scripts/blind_options.py http://127.0.0.1:8765 --out runs/blind/reasoned-board.json --states 400 \
     --conditions reasoned-board --reasoner checkpoints/base-0.8b --adapter checkpoints/reasoner-board-lora
   ```

MLX inference was deterministic in these runs: a repeated Jeff run reproduced every choice. The `reasoned-*.json` files
in this directory hold each state's reasoner text under `reasoning`.
