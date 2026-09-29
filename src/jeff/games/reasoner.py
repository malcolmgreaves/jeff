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


# Board mode: the reasoner sees only the board (`board_only`) and works each outcome out, writing its working first.
BOARD_TASK = {
    "frogger": ("Work out what each move leads to from the board alone. Each road or river row says which way it moves "
                "and whether it moves at the end of this turn. When a row moves, everything in it shifts one square that "
                "way, and what leaves one edge comes back on the other. The frog moves first; then the rows move. On a "
                "road row, the frog is hit if a car is on its square after the cars move. On a river row, the frog must "
                "land on a log; the log then carries it, and a log that carries it past the edge of the board drowns it. "
                "F marks the frog's own square: on a road row there is no car under it, and on a river row it is on a "
                "log."),
    "pacman": ("Work out what each move leads to from the maze alone. P marks Pac-Man and G a ghost. A move takes Pac-Man "
               "to the next square that way; then each ghost moves one square towards him. So a ghost on that square, or "
               "on a square next to it, catches him this turn. Also say how many steps along the corridors the nearest "
               "ghost and the nearest pellet are that way."),
}
BOARD_FORMAT = {
    "frogger": "First copy the rows the frog can reach, numbering each square by its column.",
    "pacman": ("First copy the squares within two rows and two columns of Pac-Man, numbering each square by its column "
               "and writing - for an empty square."),
}
BOARD_START = {"frogger": "Rows you can reach:\n", "pacman": "Squares near you:\n"}
BOARD_MAX_TOKENS = 640
SQUARE_NAMES = {"#": "a wall", ".": "a pellet", " ": "empty", "G": "a ghost"}


def board_only(game: frogger.Frogger | pacman.PacMan, state: dict) -> dict:
    """The judgement wording's state without its facts for each direction, so only the board is left. Frogger's rows
    also say whether they move at the end of this turn: the board does not show that for a row that moves every 2 turns,
    and without it the outcome cannot be worked out."""
    if isinstance(game, frogger.Frogger):
        return {"board_top_to_bottom": game.board(timing=True), "legend": state["legend"], "you": state["you"]}
    return {key: value for key, value in state.items() if key not in ("directions", "note")}


def board_messages(game: str, state: dict, question: dict) -> list[dict[str, str]]:
    """The board-mode prompt for one board-only state and the judgement wording's question."""
    moves = "\n".join(f"- {key}: {text}" for key, text in question["criteria"].items())
    user = (f"Game: {'Frogger' if game == 'frogger' else 'Pac-Man'}. {RULES[game]}\n\n"
            f"Game state (JSON):\n{json.dumps(state, indent=1)}\n\nMoves:\n{moves}\n\n{BOARD_TASK[game]}\n\n"
            f"{BOARD_FORMAT[game]} Then write one line for each move, in this order: {', '.join(question['criteria'])}. "
            "Each line has the move's name, a colon, how you work out what happens, then \"=>\" and what happens.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def frogger_step(game: frogger.Frogger, move: str) -> str:
    """How the outcome of one Frogger move follows from the board, then the outcome. Checked against `game.after`."""
    dr, dc = frogger.MOVES[move]
    row, col = game.row + dr, game.col + dc
    here = "your square" if (row, col) == (game.row, game.col) else f"row {row}, column {col}"
    if row == frogger.GOAL:
        seen, found = f"row {row} is the goal row", "goal"
    elif row not in game.lanes:
        seen, found = f"row {row} is grass", "safe ground"
    else:
        lane, moves = game.lanes[row], game.moves_now(row)
        way = "right" if lane.direction == 1 else "left"
        if row in frogger.ROAD:
            source = (col - lane.direction) % frogger.WIDTH if moves else col
            car = lane.cells[source]
            shown = f"{'C' if car else '.'}" + (" (your square, no car)" if (row, source) == (game.row, game.col) else "")
            seen = (f"{here}; row {row} moves {way}, so column {col} gets what is now in column {source}: {shown}" if moves
                    else f"{here}; row {row} does not move, so column {col} keeps what is there now: {shown}")
            found = "car" if car else "empty road"
        elif not lane.cells[col]:
            seen, found = f"{here} is open water", "water"
        else:
            new = col + lane.direction if moves else col
            seen = f"{here} is a log" + (f"; the log moves {way} to column {new}" if moves else "; the log does not move")
            found = "log" if 0 <= new < frogger.WIDTH else "log carrying you off the board"
            if found != "log":
                seen += ", past the edge of the board"
    if found != game.after(row, col):
        raise AssertionError(f"Frogger working says {found!r}, the game says {game.after(row, col)!r} for {move}")
    return f"{seen} => {after_label(frogger.outcome(move, found))}"


def pacman_step(game: pacman.PacMan, picture: list[str], move: str) -> str:
    """How the outcome of one Pac-Man move follows from the maze, then the outcome. Checked against `game.leaving`."""
    first = pacman.neighbours(game.pac)[move]
    around = [(way, (first[0] + dr, first[1] + dc)) for way, (dr, dc) in pacman.MOVES.items()
              if (first[0] + dr, first[1] + dc) != game.pac]
    name = lambda square: f"({square[0]}, {square[1]}) {SQUARE_NAMES[picture[square[0]][square[1]]]}"  # noqa: E731
    near = any(picture[r][c] == "G" for r, c in [first] + [square for _, square in around])
    steps = game.leaving(move)
    ghost, pellet = steps["steps_to_nearest_ghost"], steps["steps_to_nearest_pellet"]
    if near != (ghost is not None and ghost <= 2):
        raise AssertionError(f"Pac-Man working says ghost near={near}, the game says {ghost} steps for {move}")
    distances = ((f"the nearest ghost that way is {steps_away(ghost)}" if ghost is not None else "no ghost can be reached that way")
                 + (f", the nearest pellet {steps_away(pellet)}" if pellet is not None else ", no pellets that way"))
    return (f"the next square is {name(first)}; next to it: {', '.join(f'{way} {name(square)}' for way, square in around)}; "
            f"{'a ghost is within 2 steps' if near else 'no ghost within 2 steps'}; {distances} => {pacman.outcome(pellet, ghost)}")


def board_target(game: frogger.Frogger | pacman.PacMan, question: dict) -> str:
    """The lines a correct board-mode reasoner writes: the nearby squares, then the working and outcome for each move."""
    if isinstance(game, frogger.Frogger):
        lines = [BOARD_START["frogger"].rstrip("\n")]
        board = {int(line.split()[1]): line.rsplit(": ", 1)[1] for line in game.board()}
        for row in (game.row + 1, game.row, game.row - 1):
            if row in game.lanes:
                timing = (f"moves {'right' if game.lanes[row].direction == 1 else 'left'} at the end of this turn"
                          if game.moves_now(row) else "does not move this turn")
                lines.append(f"row {row} ({game.kind(row)}, {timing}): " + " ".join(f"{i}:{s}" for i, s in enumerate(board[row])))
            elif 0 <= row <= frogger.GOAL:
                lines.append(f"row {row} ({'the goal row' if row == frogger.GOAL else 'grass'})")
        return "\n".join(lines + [f"{move}: {frogger_step(game, move)}" for move in question["criteria"]])
    picture, (r, c) = game.picture(), game.pac
    lines = [BOARD_START["pacman"].rstrip("\n")]
    for row in range(max(0, r - 2), min(len(picture), r + 3)):
        lines.append(f"row {row}: " + " ".join(f"{col}:{picture[row][col].replace(' ', '-')}"
                                               for col in range(max(0, c - 2), min(len(picture[0]), c + 3))))
    return "\n".join(lines + [f"{move}: {pacman_step(game, picture, move)}" for move in question["criteria"]])


class Reasoner:
    """A chat model served in-process by MLX, optionally with a LoRA adapter; greedy decoding, no thinking block. With
    `board`, it gets board-only states and the board-mode prompt."""

    def __init__(self, model: str | Path, adapter: str | Path | None = None, board: bool = False) -> None:
        from mlx_lm import load

        self.model_path, self.adapter_path, self.board = str(model), str(adapter) if adapter else None, board
        self.model, self.tokenizer = load(self.model_path, adapter_path=self.adapter_path)

    def write(self, game: str, state: dict, question: dict, temperature: float = 0.0, seed: int | None = None) -> str:
        """The reasoner's lines. The reply is started for it (with the first move's name, or in board mode the working's
        heading), so it continues in the asked format. Decoding is greedy unless `temperature` is above 0."""
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        if self.board:
            chat, start, limit = board_messages(game, state, question), BOARD_START[game], BOARD_MAX_TOKENS
        else:
            chat, start, limit = messages(game, state, question), f"{next(iter(question['criteria']))}:", MAX_TOKENS
        prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=temperature) if temperature > 0 else None
        return start + str(generate(self.model, self.tokenizer, prompt + start, max_tokens=limit, sampler=sampler))

    def options(self, game: str, state: dict, question: dict) -> tuple[dict[str, str], str, int]:
        """Jeff's options (`jeff_options`) from the reasoner's text, the text, and how many moves it wrote an outcome for."""
        text = self.write(game, state, question)
        options, parsed = jeff_options(text, question)
        return options, text, parsed


def jeff_options(text: str, question: dict) -> tuple[dict[str, str], int]:
    """Jeff's options from a reasoner's text: each move's name, then the outcome the reasoner wrote for it. A move
    without one keeps only its name. Also returns how many moves have an outcome."""
    outcomes = parse(text, list(question["criteria"]))
    options = {key: f"{name.rstrip('.')}: {outcomes[key]}" if key in outcomes else name
               for key, name in question["criteria"].items()}
    return options, len(outcomes)
