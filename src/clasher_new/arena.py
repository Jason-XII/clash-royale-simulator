"""
I need this file for some constants that tells the battle engine information about the arena.
"""
from dataclasses import dataclass
from core import Position


def is_valid_position(pos): return 0 <= pos.x < 18 and 0 <= pos.y < 32
def is_blocked_tile(x: int, y: int) -> bool:
    """The top and bottom row has places that are not deployable"""
    if y == 0 or y == 31:
        return x <= 5 or x >= 12
    return False

walkable_cells = set()
for x in range(0, 18):
    for y in range(0, 32):
        if is_valid_position(Position(x, y)) and not is_blocked_tile(x, y):
            walkable_cells.add((x, y))
def is_walkable(pos):
    if pos.x < 0 or pos.y < 0: return False
    x, y = int(pos.x), int(pos.y)
    return (x, y) in walkable_cells


@dataclass
class TileGrid:
    BLUE_KING_TOWER = Position(9.0, 3.0)
    BLUE_LEFT_TOWER = Position(3.5, 6.5)
    BLUE_RIGHT_TOWER = Position(14.5, 6.5)
    RED_KING_TOWER = Position(9.0, 29.0)
    RED_LEFT_TOWER = Position(3.5, 25.5)
    RED_RIGHT_TOWER = Position(14.5, 25.5)
