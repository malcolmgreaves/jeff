"""How well a Jeff server chooses a move when the options do not say what each move leads to.

Every wording is asked about exactly the same game states (Frogger and Pac-Man, driven by the rule bot with a share of
random moves, so risky states come up), and each answer is scored against the game's own facts:

- outcomes: each option says what the move leads to and whether that is good or bad ("you would be hit by a car");
- situation: each option says what is where the move leads, not whether that is good or bad ("a car will be on that square");
- listed: each option is only its key ("up", "keep going"); the state lists the situation sentences under the same keys;
- judgement: each option only names the move ("Move one square up"); the facts are fields of the state, one per direction;
- board: as judgement, with those fields removed, so the facts must be worked out from the board itself;
- rule: each option says when it is the right move, in terms of the fields of the state (as the Doom harness does for Jev).

A state is contested when some legal moves are safe and some are not: only there does the choice decide whether a life
is lost this turn. Frogger: safe = the square is free once cars and logs have moved. Pac-Man: safe = no ghost within
DEADLY steps that way; a ghost that close meets Pac-Man this turn when it moves towards him, and every wording can
tell that distance apart ("deadly", "a ghost is right there", or the step count). The rule bot always picks a safe
move when there is one.

A high safe-choice rate alone does not show that a wording's facts were used: always waiting in Frogger is safe in most
contested states. So each table also shows the best fixed option (always choosing one of the game's named moves) and
each wording's fact sensitivity: how much more often it picks an option when that option is safe than when it is not
(0 for a player that always picks the same option key).

Usage: python scripts/blind_options.py URL --out runs/blind/NAME.json [--states 400] [--games frogger pacman]"""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from jeff.games import common, frogger, pacman

CONDITIONS = ("outcomes", "situation", "listed", "judgement", "board", "rule")
EPSILON = 0.25  # share of the driver's moves that are random
FROGGER_FACTS = {"up": "square_above_after_cars_and_logs_move", "stay": "your_square_after_cars_and_logs_move",
                 "left": "square_left_after_cars_and_logs_move", "right": "square_right_after_cars_and_logs_move",
                 "down": "square_below_after_cars_and_logs_move"}
PACMAN_FACTS = ("directions", "note")
DEADLY = 2  # Pac-Man steps; the outcomes wording's "deadly" boundary (pacman.outcome)
MIN_CASES = 10  # safe and unsafe states an option name needs before it counts towards fact sensitivity
MODULES = {"frogger": frogger, "pacman": pacman}


def view(game, module, condition: str) -> tuple[dict, dict, dict[str, str]]:
    """The state and question in one condition's wording, and the move each option key stands for."""
    if condition == "listed":
        state, question, moves = view(game, module, "situation")
        state = {**state, "what_each_move_leads_to": dict(question["criteria"])}
        return state, {**question, "criteria": dict.fromkeys(question["criteria"])}, moves
    version = "judgement" if condition == "board" else condition
    game.version, game.criteria = version, module.CRITERIA[version]
    state, question = game.observe()
    if condition == "board":
        facts = set(FROGGER_FACTS.values()) if module is frogger else set(PACMAN_FACTS)
        state = {key: value for key, value in state.items() if key not in facts}
    relative = module is pacman and game.criteria is None  # Pac-Man names these options "keep going", "turn left", ...
    moves = dict(game.named) if relative else {key: key for key in question["criteria"]}
    return state, question, moves


def truth(game, module) -> tuple[dict[str, bool], str]:
    """Whether each legal move is safe, and the rule bot's move."""
    state, question, _ = view(game, module, "rule")
    if module is frogger:
        safe = {move: game.safe(state[FROGGER_FACTS[move]]) for move in question["criteria"]}
    else:
        ghost = {move: state["directions"][move]["steps_to_nearest_ghost"] for move in question["criteria"]}
        safe = {move: steps is None or steps > DEADLY for move, steps in ghost.items()}
    return safe, module.rule_player(state, question)


def play(name: str, client: common.JeffClient, count: int, seed: int, steps: int) -> list[dict]:
    """Ask every condition about `count` states, over episodes seed, seed + 1, ... of at most `steps` turns each."""
    module = MODULES[name]
    game = (frogger.Frogger if module is frogger else pacman.PacMan)("rule")
    records: list[dict] = []
    episode = 0
    while len(records) < count:
        game.reset(seed + episode)
        driver = random.Random(f"{seed + episode}-driver")
        for turn in range(steps):
            if game.done or len(records) == count:
                break
            safe, rule_move = truth(game, module)
            answers = {}
            for condition in CONDITIONS:
                state, question, moves = view(game, module, condition)
                answer = client.ask(state, {"move": question})["move"]
                answers[condition] = {"choice": moves[answer["choice"]], "keys": {move: key for key, move in moves.items()},
                                      "probabilities": {moves[key]: p for key, p in answer["probabilities"].items()}}
            records.append({"seed": seed + episode, "turn": turn, "safe": safe, "rule_move": rule_move, "answers": answers})
            if len(records) % 50 == 0:
                print(f"{name}: {len(records)} / {count} states", flush=True)
            view(game, module, "rule")  # step() takes plain moves in the rule wording
            game.step(rule_move if driver.random() >= EPSILON else driver.choice(sorted(safe)))
        episode += 1
    return records


def wilson(hits: int, n: int) -> list[float]:
    """95% Wilson interval for a proportion."""
    z = 1.96
    centre = (hits + z * z / 2) / (n + z * z)
    half = z * math.sqrt(hits * (n - hits) / n + z * z / 4) / (n + z * z)
    return [centre - half, centre + half]


def paired(a: list[int], b: list[int]) -> list[float]:
    """95% normal interval for the mean paired difference a - b."""
    differences = [x - y for x, y in zip(a, b, strict=True)]
    mean, error = statistics.fmean(differences), statistics.stdev(differences) / math.sqrt(len(differences))
    return [mean - 1.96 * error, mean + 1.96 * error]


def by_name(contested: list[dict], condition: str) -> dict[str, list[tuple[dict, str]]]:
    """For each option key of a condition, the contested states where it is legal, with the move it stands for there."""
    groups: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    for r in contested:
        for move, name in r["answers"][condition]["keys"].items():
            groups[name].append((r, move))
    return groups


def sensitivity(groups: dict[str, list[tuple[dict, str]]], picks: Callable[[dict, str], float]) -> float:
    """How much more often a player picks an option when it is safe than when it is not: P(picked | safe) minus
    P(picked | unsafe), averaged over option keys with at least MIN_CASES of each, weighted by the states where the key
    is legal. `picks` gives the probability that the player picks a move in a state. A player that always picks the same
    key (such as one that always waits) scores 0."""
    total = weight = 0.0
    for cases in groups.values():
        safe = [picks(r, move) for r, move in cases if r["safe"][move]]
        unsafe = [picks(r, move) for r, move in cases if not r["safe"][move]]
        if len(safe) >= MIN_CASES and len(unsafe) >= MIN_CASES:
            total += len(cases) * (statistics.fmean(safe) - statistics.fmean(unsafe))
            weight += len(cases)
    return total / weight


def summarise(records: list[dict]) -> dict:
    contested = [r for r in records if any(r["safe"].values()) and not all(r["safe"].values())]
    groups = by_name(contested, "outcomes")  # the game's own names for the moves ("stay", "reverse", ...)
    fixed = {name: statistics.fmean(r["safe"][move] for r, move in cases)
             for name, cases in groups.items() if len(cases) >= len(contested) / 2}  # names legal in most states
    best = max(fixed, key=fixed.__getitem__)
    hits = {c: [int(r["safe"][r["answers"][c]["choice"]]) for r in contested] for c in CONDITIONS}
    agree = {c: [int(r["answers"][c]["choice"] == r["rule_move"]) for r in records] for c in CONDITIONS}
    result: dict = {"states": len(records), "contested": len(contested),
                    "chance_safe": statistics.fmean(sum(r["safe"].values()) / len(r["safe"]) for r in contested),
                    "chance_agrees": statistics.fmean(1 / len(r["safe"]) for r in records),
                    "chance_sensitivity": sensitivity(groups, lambda r, move: 1 / len(r["safe"])),
                    "best_fixed": {"option": best, "legal_in": len(groups[best]), "safe_choice": fixed[best]},
                    "rule_bot_sensitivity": sensitivity(groups, lambda r, move: float(r["rule_move"] == move)),
                    "conditions": {}}
    for c in CONDITIONS:
        result["conditions"][c] = {
            "safe_choice": statistics.fmean(hits[c]), "safe_choice_ci95": wilson(sum(hits[c]), len(contested)),
            "fact_sensitivity": sensitivity(by_name(contested, c), lambda r, move, c=c: float(r["answers"][c]["choice"] == move)),
            "safe_minus_outcomes_ci95": paired(hits[c], hits["outcomes"]) if c != "outcomes" else None,
            "probability_on_safe": statistics.fmean(sum(p for move, p in r["answers"][c]["probabilities"].items() if r["safe"][move])
                                                    for r in contested),
            "agrees_with_rule_bot": statistics.fmean(agree[c]), "agrees_ci95": wilson(sum(agree[c]), len(records))}
    return result


def table(name: str, summary: dict) -> str:
    best = summary["best_fixed"]
    lines = [f"{name}: {summary['states']} states, {summary['contested']} contested",
             "| Condition | Safe choice (contested) | 95% CI | Fact sensitivity | Probability on safe moves | Agrees with rule bot (all) |",
             "|---|---|---|---|---|---|",
             f"| random moves | {summary['chance_safe']:.1%} | | {summary['chance_sensitivity']:+.2f} | {summary['chance_safe']:.1%} | {summary['chance_agrees']:.1%} |",
             f"| always \"{best['option']}\" (legal in {best['legal_in']}) | {best['safe_choice']:.1%} | | 0.00 | | |",
             f"| rule bot | 100.0% | | {summary['rule_bot_sensitivity']:+.2f} | | 100.0% |"]
    for c, s in summary["conditions"].items():
        low, high = s["safe_choice_ci95"]
        lines.append(f"| {c} | {s['safe_choice']:.1%} | {low:.1%}-{high:.1%} | {s['fact_sensitivity']:+.2f} | "
                     f"{s['probability_on_safe']:.1%} | {s['agrees_with_rule_bot']:.1%} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", nargs="+", choices=sorted(MODULES), default=sorted(MODULES))
    parser.add_argument("--states", type=int, default=400, help="States per game")
    parser.add_argument("--steps", type=int, default=60, help="Turns per episode before the next seed starts")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    client = common.JeffClient(args.url)
    records = {name: play(name, client, args.states, args.seed, args.steps) for name in args.games}
    summaries = {name: summarise(rows) for name, rows in records.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"url": args.url, "server": client.http.get(args.url.rstrip("/") + "/health").json(),
                                    "created": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                                    "epsilon": EPSILON, "steps": args.steps, "summary": summaries, "records": records}, indent=1) + "\n")
    print("\n\n".join(table(name, summary) for name, summary in summaries.items()))


if __name__ == "__main__":
    main()
