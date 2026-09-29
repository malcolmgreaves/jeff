"""A generative model that reads a game state and writes what each move leads to, so that Jeff can choose between moves
whose options state those outcomes.

The reasoner sees the full state the harness builds for its judgement wording (the board, plus the facts for each
direction) and the legal moves. It writes one line per move: what the state shows for that move, then what happens.
The outcome part of each line becomes that move's option text for Jeff. `target` writes the same lines from the game's
own facts, in the outcomes wording's words, as fine-tuning data for the reasoner."""

import json
import re
from pathlib import Path

from jeff.games import frogger, pacman

SYSTEM = ("You analyse one turn of a game. For each move the player can make, you work out from the game state what that "
          "move leads to. You do not choose a move.")
RULES = {
    "frogger": ("The frog starts on the bottom row and must reach the top row, the goal. Each turn the frog moves one square "
                "up, down, left or right, or stays where it is; then the cars and logs move. On a road row, a car on the "
                "frog's square hits it and it loses a life. On a river row, the frog must be on a log, or it drowns and "
                "loses a life; a log that carries the frog off the board also drowns it. Reaching the goal row scores a "
                "crossing."),
    "pacman": ("Pac-Man moves through a maze eating pellets while two ghosts chase him. Each turn Pac-Man moves one square "
               "along a corridor; then each ghost moves one square, usually towards Pac-Man. If a ghost reaches Pac-Man's "
               "square, or they pass through each other, Pac-Man loses a life. Eating every pellet clears the maze."),
}
TASK = {
    "frogger": ("The fields ending in _after_cars_and_logs_move say what will be on each square the frog can reach, once the "
                "cars and logs have moved. For each move, look at what will be on the square that move leads to, and work "
                "out what happens to the frog: whether it loses a life, and whether it gets closer to the goal."),
    "pacman": ("The directions field gives, for each move, the steps along the corridors to the nearest ghost and to the "
               "nearest pellet when Pac-Man leaves that way. Moving towards a ghost shortens the distance by one step, and "
               "the ghost then moves one step closer, so the distance can shrink by two in one turn. For each move, work "
               "out what happens: whether Pac-Man runs into a ghost and loses a life this turn, whether a ghost is close "
               "enough to be a danger soon, and whether he eats a pellet soon."),
}
EXAMPLE = {
    "frogger": ("For example, if the game had a move named jump and a car will be on the square the frog jumps to, the "
                "line for jump would be:\njump: a car will be on that square => the frog is hit by the car and loses a life"),
    "pacman": ("For example, if the game had a move named jump, with the nearest ghost 9 steps away and the nearest pellet "
               "1 step away that way, the line for jump would be:\njump: nearest ghost 9 steps away, nearest pellet 1 step "
               "away => no ghost can reach Pac-Man soon, and he eats a pellet right away"),
}
MAX_TOKENS = 400
# Frogger's state field for each move
FACTS = {"up": "square_above_after_cars_and_logs_move", "stay": "your_square_after_cars_and_logs_move",
         "left": "square_left_after_cars_and_logs_move", "right": "square_right_after_cars_and_logs_move",
         "down": "square_below_after_cars_and_logs_move"}
# A line that starts a move's answer: "up: ...", "- **up**: ...", or "<move>: up: ..." (the placeholder copied).
START = re.compile(r"^[\s*_`#>-]*(?:<move>\s*:?\s*)?([A-Za-z]+)[\s*_`]*:\s*(.*)$")


def messages(game: str, state: dict, question: dict) -> list[dict[str, str]]:
    """The reasoner's prompt for one state; `question` is the judgement wording's question (options name the moves)."""
    moves = "\n".join(f"- {key}: {text}" for key, text in question["criteria"].items())
    user = (f"Game: {'Frogger' if game == 'frogger' else 'Pac-Man'}. {RULES[game]}\n\n"
            f"Game state (JSON):\n{json.dumps(state, indent=1)}\n\nMoves:\n{moves}\n\n{TASK[game]}\n\n"
            "Answer with one line for each move: the move's name, a colon, what the state shows for that move, then "
            f"\"=>\" and what happens. {EXAMPLE[game]}\n\n"
            f"Write one line for each of these moves, in this order: {', '.join(question['criteria'])}. "
            "Keep each line under 30 words.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text: str, keys: list[str]) -> dict[str, str]:
    """The outcome the reasoner wrote for each move: the part after the last "=>" in that move's answer, or the whole
    answer without one. An answer runs from the line that starts with the move's name to the next such line. A move the
    reasoner wrote nothing for is left out; a move answered twice keeps its first answer."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = START.match(line)
        key = match.group(1).lower() if match else None
        if match and key in keys:
            current = None if key in blocks else key
            if current is not None:
                blocks[current] = [match.group(2).strip()]
        elif current is not None and line.strip():
            blocks[current].append(line.strip())
    outcomes = {key: " ".join(lines).split("=>")[-1].strip() for key, lines in blocks.items()}
    return {key: outcomes[key] for key in keys if outcomes.get(key)}


def after_label(text: str) -> str:
    """ "Hop forward: onto a log." -> "onto a log." """
    return text.split(": ", 1)[1]


def steps_away(steps: int) -> str:
    return f"{steps} step{'' if steps == 1 else 's'} away"


def target(game: str, state: dict, question: dict) -> str:
    """The lines a correct reasoner writes for this judgement-wording state, from the game's own facts."""
    lines = []
    for move in question["criteria"]:
        if game == "frogger":
            what = state[FACTS[move]]
            lines.append(f"{move}: {after_label(frogger.situation(move, what))} => {after_label(frogger.outcome(move, what))}")
        else:
            steps = state["directions"][move]
            ghost, pellet = steps["steps_to_nearest_ghost"], steps["steps_to_nearest_pellet"]
            seen = (f"the nearest ghost that way is {steps_away(ghost)}" if ghost is not None else "no ghost can be reached that way") + \
                   (f", the nearest pellet {steps_away(pellet)}" if pellet is not None else ", no pellets that way")
            lines.append(f"{move}: {seen} => {pacman.outcome(pellet, ghost)}")
    return "\n".join(lines)


class Reasoner:
    """A chat model served in-process by MLX, optionally with a LoRA adapter; greedy decoding, no thinking block."""

    def __init__(self, model: str | Path, adapter: str | Path | None = None) -> None:
        from mlx_lm import load

        self.model_path, self.adapter_path = str(model), str(adapter) if adapter else None
        self.model, self.tokenizer = load(self.model_path, adapter_path=self.adapter_path)

    def write(self, game: str, state: dict, question: dict) -> str:
        """The reasoner's lines; the reply is started with the first move's name, so it continues in the asked format."""
        from mlx_lm import generate

        start = f"{next(iter(question['criteria']))}:"
        prompt = self.tokenizer.apply_chat_template(messages(game, state, question), add_generation_prompt=True,
                                                    enable_thinking=False, tokenize=False)
        return start + str(generate(self.model, self.tokenizer, prompt + start, max_tokens=MAX_TOKENS))

    def options(self, game: str, state: dict, question: dict) -> tuple[dict[str, str], str, int]:
        """Jeff's options (each move's name, then the outcome the reasoner wrote for it), the reasoner's text, and how
        many moves it wrote an outcome for. A move without one keeps only its name."""
        text = self.write(game, state, question)
        outcomes = parse(text, list(question["criteria"]))
        options = {key: f"{name.rstrip('.')}: {outcomes[key]}" if key in outcomes else name
                   for key, name in question["criteria"].items()}
        return options, text, len(outcomes)
