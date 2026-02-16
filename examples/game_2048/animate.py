from __future__ import annotations

# Animation utilities to render 2048 board sequences to a smooth GIF.
#
# Primary API:
#     animate_2048_boards(boards, out_path, actions=None, ...)
#
# Features:
# - Smooth slide animations for tiles (with merge pop) when actions are provided.
# - Graceful fallback to crossfade animation when actions are unknown.
# - New tile spawn "pop" animation.
#
# Inputs:
# - boards: list of numpy 4x4 int arrays using exponent encoding (0=empty, 1->2, 2->4, ...)
# - actions (optional): list of Action enums parallel to transitions between boards
#
# Outputs:
# - A GIF file saved to `out_path`
#
# Dependencies: moviepy (optional), numpy, Pillow

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

from enum import Enum


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# ------------------------------- Drawing utils -------------------------------


def _tile_color(power: int) -> Tuple[int, int, int]:
    """Return an RGB color for a given tile exponent (0 means empty)."""
    # 2048-like palette (approximate)
    palette = {
        0: (205, 193, 180),  # empty cell color
        1: (238, 228, 218),  # 2
        2: (237, 224, 200),  # 4
        3: (242, 177, 121),  # 8
        4: (245, 149, 99),  # 16
        5: (246, 124, 95),  # 32
        6: (246, 94, 59),  # 64
        7: (237, 207, 114),  # 128
        8: (237, 204, 97),  # 256
        9: (237, 200, 80),  # 512
        10: (237, 197, 63),  # 1024
        11: (237, 194, 46),  # 2048
    }
    return palette.get(power, (60, 58, 50))  # beyond 2048


def _text_color(power: int) -> Tuple[int, int, int]:
    # Dark text for small numbers, white for big
    if power <= 2:
        return (119, 110, 101)
    return (249, 246, 242)


def _rounded_rectangle(draw: ImageDraw.ImageDraw, xy, radius, fill):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)


def _draw_board_base(img: Image.Image, grid_size: int, cell: int, margin: int):
    draw = ImageDraw.Draw(img, "RGBA")
    bg_color = (187, 173, 160)
    _rounded_rectangle(
        draw,
        (0, 0, img.width, img.height),
        radius=margin,
        fill=bg_color,
    )
    # The empty cells grid
    for r in range(grid_size):
        for c in range(grid_size):
            x = margin + c * (cell + margin)
            y = margin + r * (cell + margin)
            _rounded_rectangle(
                draw,
                (x, y, x + cell, y + cell),
                radius=int(cell * 0.1),
                fill=_tile_color(0),
            )


def _draw_tile(
    draw: ImageDraw.ImageDraw,
    value_power: int,
    x: float,
    y: float,
    size: float,
    font: ImageFont.FreeTypeFont,
    alpha: float = 1.0,
):
    # Draw a single tile at floating coords with size and alpha.
    color = _tile_color(value_power)
    fill = (*color, int(255 * np.clip(alpha, 0, 1)))
    radius = int(size * 0.1)
    _rounded_rectangle(
        draw,
        (x, y, x + size, y + size),
        radius=radius,
        fill=fill,
    )
    if value_power > 0:
        value = 2 ** int(value_power)
        text = str(value)
        # Try to fit text: scale down font size for larger numbers
        # Render with the provided font; if too big, Pillow will clip slightly but acceptable
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        tx = x + (size - tw) / 2
        ty = y + (size - th) / 2 - 2  # slightly nudge up
        draw.text((tx, ty), text, fill=_text_color(value_power), font=font)


# -------------------------- Movement mapping helpers -------------------------


@dataclass
class MovePiece:
    sources: List[Tuple[int, int, int]]  # list of (r, c, power)
    dest: Tuple[int, int]  # (r, c)
    merged: bool  # whether multiple sources merged
    result_power: int  # power at destination after merge or same power if no merge


def _map_row_move_left(
    row: np.ndarray,
) -> List[Tuple[List[Tuple[int, int]], Tuple[int, int], bool, int]]:
    """
    Map a single row move to the left into movements.
    Returns a list of tuples: (source_cells, dest_cell, merged, result_power)
        - source_cells: list of (r, c) pairs (same row, distinct columns)
        - dest_cell: (r, c)
        - merged: True if two identical tiles merged into one
        - result_power: exponent at destination after move/merge
    """
    non_zero = [(j, row[j]) for j in range(4) if row[j] != 0]
    out = []
    k = 0  # destination col index
    i = 0
    while i < len(non_zero):
        j, p = non_zero[i]
        if i + 1 < len(non_zero) and non_zero[i + 1][1] == p:
            # merge j and j+1 into col k, power+1
            out.append(
                ([(None, j), (None, non_zero[i + 1][0])], (None, k), True, int(p + 1))
            )
            i += 2
            k += 1
        else:
            out.append(([(None, j)], (None, k), False, int(p)))
            i += 1
            k += 1
    return out


def _build_move_mapping(prev: np.ndarray, action: Action) -> List[MovePiece]:
    """Build mapping of moving pieces from prev board to board after applying action.

    We do not include the random new tile here; this maps only the slide/merge.
    """
    pieces: List[MovePiece] = []
    if action in (Action.LEFT, Action.RIGHT):
        for r in range(4):
            row = prev[r]
            # Build index-power pairs for non-zero tiles
            nz = [(j, int(row[j])) for j in range(4) if row[j] != 0]
            if action == Action.LEFT:
                i = 0
                k = 0
                while i < len(nz):
                    j, p = nz[i]
                    if i + 1 < len(nz) and nz[i + 1][1] == p:
                        j2, _ = nz[i + 1]
                        pieces.append(
                            MovePiece(
                                sources=[(r, j, p), (r, j2, p)],
                                dest=(r, k),
                                merged=True,
                                result_power=p + 1,
                            )
                        )
                        i += 2
                        k += 1
                    else:
                        pieces.append(
                            MovePiece(
                                sources=[(r, j, p)],
                                dest=(r, k),
                                merged=False,
                                result_power=p,
                            )
                        )
                        i += 1
                        k += 1
            else:  # RIGHT
                nz_rev = [(3 - j, int(row[j])) for j in range(4) if row[j] != 0][::-1]
                i = 0
                k = 0
                while i < len(nz_rev):
                    j, p = nz_rev[i]
                    if i + 1 < len(nz_rev) and nz_rev[i + 1][1] == p:
                        j2, _ = nz_rev[i + 1]
                        dest_c = 3 - k
                        pieces.append(
                            MovePiece(
                                sources=[(r, 3 - j, p), (r, 3 - j2, p)],
                                dest=(r, dest_c),
                                merged=True,
                                result_power=p + 1,
                            )
                        )
                        i += 2
                        k += 1
                    else:
                        dest_c = 3 - k
                        pieces.append(
                            MovePiece(
                                sources=[(r, 3 - j, p)],
                                dest=(r, dest_c),
                                merged=False,
                                result_power=p,
                            )
                        )
                        i += 1
                        k += 1
    else:  # UP or DOWN act on columns
        for c in range(4):
            col = prev[:, c]
            nz = [(i, int(col[i])) for i in range(4) if col[i] != 0]
            if action == Action.UP:
                i = 0
                k = 0
                while i < len(nz):
                    r, p = nz[i]
                    if i + 1 < len(nz) and nz[i + 1][1] == p:
                        r2, _ = nz[i + 1]
                        pieces.append(
                            MovePiece(
                                sources=[(r, c, p), (r2, c, p)],
                                dest=(k, c),
                                merged=True,
                                result_power=p + 1,
                            )
                        )
                        i += 2
                        k += 1
                    else:
                        pieces.append(
                            MovePiece(
                                sources=[(r, c, p)],
                                dest=(k, c),
                                merged=False,
                                result_power=p,
                            )
                        )
                        i += 1
                        k += 1
            else:  # DOWN
                nz_rev = [(3 - i, int(col[i])) for i in range(4) if col[i] != 0][::-1]
                i = 0
                k = 0
                while i < len(nz_rev):
                    r, p = nz_rev[i]
                    if i + 1 < len(nz_rev) and nz_rev[i + 1][1] == p:
                        r2, _ = nz_rev[i + 1]
                        dest_r = 3 - k
                        pieces.append(
                            MovePiece(
                                sources=[(3 - r, c, p), (3 - r2, c, p)],
                                dest=(dest_r, c),
                                merged=True,
                                result_power=p + 1,
                            )
                        )
                        i += 2
                        k += 1
                    else:
                        dest_r = 3 - k
                        pieces.append(
                            MovePiece(
                                sources=[(3 - r, c, p)],
                                dest=(dest_r, c),
                                merged=False,
                                result_power=p,
                            )
                        )
                        i += 1
                        k += 1
    return pieces


def _merge_line_left(line: np.ndarray) -> np.ndarray:
    non_zero = [int(x) for x in line if x != 0]
    merged: List[int] = []
    i = 0
    while i < len(non_zero):
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            merged.append(non_zero[i] + 1)
            i += 2
        else:
            merged.append(non_zero[i])
            i += 1
    out = np.zeros_like(line)
    out[: len(merged)] = np.array(merged, dtype=out.dtype)
    return out


def _apply_action_without_spawn(board: np.ndarray, action: Action) -> np.ndarray:
    b = board.copy()
    if action == Action.LEFT:
        for r in range(4):
            b[r] = _merge_line_left(b[r])
    elif action == Action.RIGHT:
        for r in range(4):
            b[r] = _merge_line_left(b[r][::-1])[::-1]
    elif action == Action.UP:
        b = b.T
        for r in range(4):
            b[r] = _merge_line_left(b[r])
        b = b.T
    elif action == Action.DOWN:
        b = b.T
        for r in range(4):
            b[r] = _merge_line_left(b[r][::-1])[::-1]
        b = b.T
    return b


def _find_spawn_tile(
    after_move: np.ndarray, next_board: np.ndarray
) -> Optional[Tuple[int, int, int]]:
    """Find the spawned tile by comparing the board after move (no spawn) and the next board.
    Returns (r, c, power) or None if not found.
    """
    diff_mask = (after_move == 0) & (next_board > 0)
    candidates = np.argwhere(diff_mask)
    for r, c in candidates:
        p = int(next_board[r, c])
        if p in (1, 2):  # spawn is 2 or 4 in exponent encoding
            return (int(r), int(c), p)
    # Fallback: if exactly one cell differs and increased from 0
    if candidates.shape[0] == 1:
        r, c = candidates[0]
        return (int(r), int(c), int(next_board[r, c]))
    return None


# ------------------------------- Main animator -------------------------------


def animate_2048_boards(
    boards: List[np.ndarray],
    out_path: str,
    actions: Optional[List[Action]] = None,
    fps: int = 30,
    move_duration: float = 0.45,
    spawn_duration: float = 0.18,
    hold_duration: float = 0.25,
    tile_size: int = 96,
    margin: int = 12,
    font_path: Optional[str] = None,
    crossfade_when_no_actions: bool = True,
):
    """
    Animate a sequence of 2048 boards to a GIF saved at `out_path`.

    - If `actions` are provided (len = len(boards)-1), tiles will slide and merge smoothly.
    - If `actions` is None, a smooth crossfade is used between boards.
    - New tile spawn is animated as a quick pop.
    """
    assert len(boards) >= 2, "Need at least two boards to animate"
    grid_size = 4
    W = margin + grid_size * (tile_size + margin)
    H = W

    # Choose a font size relative to tile size
    font_size = int(tile_size * 0.42)
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Precompute per-transition metadata
    transitions = []
    total_duration = 0.0
    for i in range(len(boards) - 1):
        prev = boards[i].astype(int)
        nxt = boards[i + 1].astype(int)
        step = {
            "prev": prev,
            "next": nxt,
            "action": None,
            "pieces": None,
            "after_move": None,
            "spawn": None,
            "t_start": total_duration,
            "t_move": move_duration,
            "t_spawn": spawn_duration,
            "t_hold": hold_duration,
        }
        if actions is not None:
            act = actions[i]
            step["action"] = act
            pieces = _build_move_mapping(prev, act)
            after_move = _apply_action_without_spawn(prev, act)
            spawn = _find_spawn_tile(after_move, nxt)
            step["pieces"] = pieces
            step["after_move"] = after_move
            step["spawn"] = spawn
        transitions.append(step)
        total_duration += move_duration + spawn_duration + hold_duration

    # Pre-draw static grid background
    base_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    _draw_board_base(base_img, grid_size, tile_size, margin)

    def cell_to_xy(r: float, c: float, size: float = tile_size) -> Tuple[float, float]:
        x = margin + c * (tile_size + margin)
        y = margin + r * (tile_size + margin)
        return x, y

    def draw_tiles_on(
        img: Image.Image, tiles: List[Tuple[float, float, float, float, int, float]]
    ):
        """Draw tiles on img.
        tiles: list of (r, c, off_r, off_c, power, alpha) with r,c as floats in cell units.
        """
        draw = ImageDraw.Draw(img, "RGBA")
        for r, c, orf, ocf, p, a in tiles:
            x, y = cell_to_xy(r + orf, c + ocf)
            _draw_tile(draw, p, x, y, tile_size, font, alpha=a)

    def render_board(nparr: np.ndarray) -> Image.Image:
        img = base_img.copy()
        tiles = []
        for r in range(4):
            for c in range(4):
                p = int(nparr[r, c])
                if p > 0:
                    tiles.append((r, c, 0.0, 0.0, p, 1.0))
        draw_tiles_on(img, tiles)
        return img

    def make_frame(t: float):
        # Determine which transition and local time
        if t >= total_duration:
            # Return last board
            return np.array(render_board(boards[-1]).convert("RGB"))

        # Find the step for this t
        step_idx = 0
        for idx, step in enumerate(transitions):
            if (
                step["t_start"]
                <= t
                < step["t_start"] + step["t_move"] + step["t_spawn"] + step["t_hold"]
            ):
                step_idx = idx
                break

        step = transitions[step_idx]
        local_t = t - step["t_start"]
        t_move = step["t_move"]
        t_spawn = step["t_spawn"]
        t_hold = step["t_hold"]
        prev = step["prev"]
        nxt = step["next"]

        # Segment 1: movement (0..t_move)
        if (
            local_t < t_move
            and step["action"] is not None
            and step["pieces"] is not None
        ):
            # If nothing moves for this action, just render the previous board
            if len(step["pieces"]) == 0:
                return np.array(render_board(prev).convert("RGB"))

            img = base_img.copy()
            draw = ImageDraw.Draw(img, "RGBA")
            u = max(0.0, min(1.0, local_t / t_move))

            # Draw stationary tiles first (those not part of any moving piece)
            moving_set = {
                (sr, sc) for piece in step["pieces"] for (sr, sc, _p) in piece.sources
            }
            for rr in range(4):
                for cc in range(4):
                    p = int(prev[rr, cc])
                    if p > 0 and (rr, cc) not in moving_set:
                        _draw_tile(
                            draw, p, *cell_to_xy(rr, cc), tile_size, font, alpha=1.0
                        )

            # For merged tiles, swap to merged result near the end
            merge_switch = 0.7

            # Draw moving pieces on top
            for piece in step["pieces"]:
                dest_r, dest_c = piece.dest
                if not piece.merged:
                    # single source
                    sr, sc, sp = piece.sources[0]
                    r = sr + (dest_r - sr) * u
                    c = sc + (dest_c - sc) * u
                    _draw_tile(draw, sp, *cell_to_xy(r, c), tile_size, font, alpha=1.0)
                else:
                    # two sources converge, then pop merged
                    (r1, c1, p1), (r2, c2, p2) = piece.sources
                    if u < merge_switch:
                        rr1 = r1 + (dest_r - r1) * u / merge_switch
                        cc1 = c1 + (dest_c - c1) * u / merge_switch
                        rr2 = r2 + (dest_r - r2) * u / merge_switch
                        cc2 = c2 + (dest_c - c2) * u / merge_switch
                        _draw_tile(
                            draw, p1, *cell_to_xy(rr1, cc1), tile_size, font, alpha=1.0
                        )
                        _draw_tile(
                            draw, p2, *cell_to_xy(rr2, cc2), tile_size, font, alpha=1.0
                        )
                    else:
                        # draw the merged tile with a slight scale pop
                        v = (u - merge_switch) / (1 - merge_switch)
                        scale = 1.0 + 0.15 * np.sin(np.pi * min(v, 1.0))
                        size = tile_size * scale
                        x, y = cell_to_xy(dest_r, dest_c)
                        x -= (size - tile_size) / 2
                        y -= (size - tile_size) / 2
                        _rounded_rectangle(
                            draw,
                            (x, y, x + size, y + size),
                            radius=int(size * 0.1),
                            fill=(*_tile_color(piece.result_power), 255),
                        )
                        # Label
                        value = 2 ** int(piece.result_power)
                        text = str(value)
                        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
                        tx = x + (size - tw) / 2
                        ty = y + (size - th) / 2 - 2
                        draw.text(
                            (tx, ty),
                            text,
                            fill=_text_color(piece.result_power),
                            font=font,
                        )

            return np.array(img.convert("RGB"))

        # Segment 2: spawn (t_move..t_move+t_spawn)
        if (
            local_t < t_move + t_spawn
            and step["action"] is not None
            and step["after_move"] is not None
        ):
            u = max(0.0, min(1.0, (local_t - t_move) / max(t_spawn, 1e-6)))
            base = render_board(step["after_move"]).copy()
            draw = ImageDraw.Draw(base, "RGBA")
            if step["spawn"] is not None:
                sr, sc, sp = step["spawn"]
                # pop from small to full size
                scale = 0.3 + 0.7 * u
                size = tile_size * scale
                x, y = cell_to_xy(sr, sc)
                x -= (size - tile_size) / 2
                y -= (size - tile_size) / 2
                _rounded_rectangle(
                    draw,
                    (x, y, x + size, y + size),
                    radius=int(size * 0.1),
                    fill=(*_tile_color(sp), int(255 * u)),
                )
                value = 2 ** int(sp)
                text = str(value)
                tw, th = draw.textbbox((0, 0), text, font=font)[2:]
                tx = x + (size - tw) / 2
                ty = y + (size - th) / 2 - 2
                draw.text((tx, ty), text, fill=_text_color(sp), font=font)
            return np.array(base.convert("RGB"))

        # Segment 3: hold next board
        if step["action"] is not None:
            return np.array(render_board(nxt).convert("RGB"))

        # No actions provided: crossfade between boards for smoothness
        if crossfade_when_no_actions:
            total = t_move + t_spawn + t_hold
            u = max(0.0, min(1.0, local_t / total))
            A = render_board(prev).convert("RGBA")
            B = render_board(nxt).convert("RGBA")
            out = Image.blend(A, B, u)
            return np.array(out.convert("RGB"))
        else:
            # Fallback to just next board
            return np.array(render_board(nxt).convert("RGB"))

    # Write out GIF using moviepy if available, else imageio directly
    try:
        from moviepy.editor import VideoClip  # type: ignore

        clip = VideoClip(make_frame, duration=total_duration)
        clip.write_gif(out_path, fps=fps, program="imageio")
    except Exception:
        # Sample all frames at given fps
        n_frames = int(np.ceil(total_duration * fps))
        frames = []
        for i in range(n_frames + 1):
            t = min(total_duration, i / fps)
            frames.append(make_frame(t))
        imageio.mimsave(out_path, frames, duration=1 / fps)


# ------------------------------ Quick demo CLI -------------------------------

if __name__ == "__main__":
    import os
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python animate.py <path_to_boards.npz>")
        print(
            "The .npz file should contain 'boards' (list of 4x4 numpy arrays) and optionally 'actions'."
        )
        sys.exit(1)

    extra = np.load(path, allow_pickle=True)
    boards, actions = extra["boards"], extra["actions"]

    out = f"{os.path.dirname(path)}/sequence.gif"
    animate_2048_boards(boards, out, actions=actions, fps=6)
    print(f"Saved demo GIF to {out}")
