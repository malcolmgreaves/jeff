"""Jeff training data in the fine-tuned reasoner's own words, for scripts/jeff_lora.py.

Game states come from the driver of scripts/blind_options.py (the rule bot with a share of random moves), on seeds no
other run uses: train from 300,000 on, dev from 400,000 on, and calibration from 500,000 on. For each state the reasoner
(jeff.games.reasoner) writes what each move leads to, and code checks every outcome it wrote against the game's own
(jeff.games.reasoner.target). With `--input board` the reasoner sees only the board (board mode), and the game's
outcomes come from jeff.games.reasoner.board_target. A state is kept only when every outcome is right. After a wrong greedy answer
the reasoner writes again with sampling, up to TRIES answers in all, and a state it never gets right is dropped. The
label always comes from the game (the rule bot's move); nothing the reasoner writes changes it.

Each row is a jeff Example with the question the reasoned condition of scripts/blind_options.py asks: the outcomes
wording's short state, and a choice between the moves, each named and followed by the reasoner's outcome.

Usage: python scripts/reasoned_jeff_data.py --reasoner MODEL_DIR --adapter ADAPTER_DIR --out data/reasoned-jeff
       [--input facts|board] [--train 1000] [--dev 100] [--calibration 100]   (states per game)"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from jeff.data import validate, write_rows
from jeff.games import frogger, pacman, reasoner
from jeff.types import Example

SEEDS = {"train": 300_000, "dev": 400_000, "calibration": 500_000}
EPSILON, STEPS = 0.25, 60  # as scripts/blind_options.py
TRIES, SAMPLE_TEMPERATURE = 4, 0.7


def observe(game, module, version: str) -> tuple[dict, dict]:
    game.version, game.criteria = version, module.CRITERIA[version]
    return game.observe()


def correct(text: str, target: str, keys: list[str]) -> bool:
    """Whether the reasoner wrote the game's own outcome for every move."""
    written, wanted = reasoner.parse(text, keys), reasoner.parse(target, keys)
    return all(written.get(key) == wanted[key] for key in keys)


def rows(name: str, count: int, seed: int, model: reasoner.Reasoner, stats: Counter[str]) -> list[Example]:
    module = frogger if name == "frogger" else pacman
    game = (frogger.Frogger if module is frogger else pacman.PacMan)("rule")
    result: list[Example] = []
    episode = 0
    while len(result) < count:
        game.reset(seed + episode)
        driver = random.Random(f"{seed + episode}-driver")
        for turn in range(STEPS):
            if game.done or len(result) == count:
                break
            short, _ = observe(game, module, "outcomes")
            full, question = observe(game, module, "judgement")
            keys = list(question["criteria"])
            if model.board:
                seen, target = reasoner.board_only(game, full), reasoner.board_target(game, question)
            else:
                seen, target = full, reasoner.target(name, full, question)
            rule_state, rule_question = observe(game, module, "rule")
            label = module.rule_player(rule_state, rule_question)
            text: str | None = None
            for tries in range(1, TRIES + 1):
                written = model.write(name, seen, question, temperature=0.0 if tries == 1 else SAMPLE_TEMPERATURE,
                                      seed=None if tries == 1 else (seed + episode) * 1000 + turn * 10 + tries)
                if correct(written, target, keys):
                    text = written
                    break
            stats["states"] += 1
            stats["kept on the first answer" if tries == 1 and text else "kept after a retry" if text else "dropped"] += 1
            if text is not None:
                options, _ = reasoner.jeff_options(text, question)
                result.append({"id": f"reasoned-{name}-{seed + episode}-{turn}", "suite": "reasoned-games",
                               "family": f"reasoned-{name}-{seed + episode}", "state": short,
                               "question": {**question, "criteria": options}, "label": label, "target": label,
                               "source": {"dataset": "reasoned-games", "game": name, "seed": seed + episode, "turn": turn,
                                          "answers_written": tries, "reasoner_input": "board" if model.board else "facts",
                                          "reasoning": text}})
                if len(result) % 100 == 0:
                    print(f"{name}: {len(result)} / {count} rows; {dict(stats)}", flush=True)
            game_move = label if driver.random() >= EPSILON else driver.choice(sorted(rule_question["criteria"]))
            observe(game, module, "rule")  # step() takes plain moves in the rule wording
            game.step(game_move)
        episode += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reasoner", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input", choices=("facts", "board"), default="facts", help="What the reasoner sees")
    parser.add_argument("--train", type=int, default=1000, help="States per game")
    parser.add_argument("--dev", type=int, default=100, help="States per game")
    parser.add_argument("--calibration", type=int, default=100, help="States per game")
    args = parser.parse_args()
    model = reasoner.Reasoner(args.reasoner, args.adapter, board=args.input == "board")
    report: dict[str, object] = {"reasoner": args.reasoner, "adapter": args.adapter, "input": args.input, "tries": TRIES,
                                 "sample_temperature": SAMPLE_TEMPERATURE, "seeds": SEEDS}
    for split, count in (("train", args.train), ("dev", args.dev), ("calibration", args.calibration)):
        stats: dict[str, dict[str, int]] = {}
        data: list[Example] = []
        for name in ("frogger", "pacman"):
            counts: Counter[str] = Counter()
            data += rows(name, count, SEEDS[split], model, counts)
            stats[name] = dict(counts)
        random.Random(f"{SEEDS[split]}-shuffle").shuffle(data)
        validate(data)
        report[split] = {"rows": len(data), "sha256": write_rows(args.out / f"{split}.jsonl", data), "states": stats}
        print(json.dumps({split: report[split]}), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
