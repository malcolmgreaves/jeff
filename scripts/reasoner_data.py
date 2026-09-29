"""Fine-tuning data for the reasoner (jeff.games.reasoner): the reasoner's prompt for a game state, and as the answer the
lines a correct reasoner writes, built from the game's own facts in the outcomes wording's words.

States come from the same driver as scripts/blind_options.py (the rule bot with a share of random moves), on seeds
disjoint from its evaluation seeds (1234 on): train from TRAIN_SEED on, valid from VALID_SEED on. The files are in
mlx-lm's chat format, for `mlx_lm.lora --data DIR --mask-prompt`.

Usage: python scripts/reasoner_data.py --out data/reasoner [--train 2000] [--valid 100]"""

import argparse
import json
import random
from pathlib import Path

from jeff.games import frogger, pacman, reasoner

TRAIN_SEED, VALID_SEED = 100_000, 200_000
EPSILON, STEPS = 0.25, 60  # as scripts/blind_options.py


def observe(game, module, version: str) -> tuple[dict, dict]:
    game.version, game.criteria = version, module.CRITERIA[version]
    return game.observe()


def rows(name: str, count: int, seed: int) -> list[dict]:
    module = frogger if name == "frogger" else pacman
    game = (frogger.Frogger if module is frogger else pacman.PacMan)("rule")
    result: list[dict] = []
    episode = 0
    while len(result) < count:
        game.reset(seed + episode)
        driver = random.Random(f"{seed + episode}-driver")
        for _ in range(STEPS):
            if game.done or len(result) == count:
                break
            state, question = observe(game, module, "judgement")
            result.append({"messages": [*reasoner.messages(name, state, question),
                                        {"role": "assistant", "content": reasoner.target(name, state, question)}]})
            rule_state, rule_question = observe(game, module, "rule")
            move = module.rule_player(rule_state, rule_question)
            game.step(move if driver.random() >= EPSILON else driver.choice(sorted(rule_question["criteria"])))
        episode += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=2000, help="States per game")
    parser.add_argument("--valid", type=int, default=100, help="States per game")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split, count, seed in (("train", args.train, TRAIN_SEED), ("valid", args.valid, VALID_SEED)):
        data = [row for name in ("frogger", "pacman") for row in rows(name, count, seed)]
        random.Random(f"{seed}-shuffle").shuffle(data)
        path = args.out / f"{split}.jsonl"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data))
        print(f"{path}: {len(data)} rows")


if __name__ == "__main__":
    main()
