from pathlib import Path
import math
from core import Position
import heapq

grid_path = Path(__file__).with_name('tilemap_lane_grid.txt')
with grid_path.open('r') as f:
    contents = [list(each) for each in f.read().splitlines()]

cell_cache = {}
neighbor_cache = {}

def position_to_cell(position: Position):
    x, y = position.x, position.y
    return math.floor(2*x), math.floor(2*y)

def cell_to_position(cell):
    if cell not in cell_cache:
        x, y = cell
        cell_cache[cell] = Position((x+0.5)/2, (y+0.5)/2)
    return cell_cache[cell]

def get_neighboring_points(x, y):
    if (x, y) in neighbor_cache: return neighbor_cache[(x,y)]
    result = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            new_x, new_y = x+dx, y+dy
            if dx == dy == 0: continue
            if new_x < 0 or new_y < 0 or new_x >= 36 or new_y >= 64: continue
            result.append((new_x, new_y))
    neighbor_cache[(x,y)] = result
    return result


# Per-cell facts about the 36x64 half-tile grid that never change, computed once.
from arena import is_walkable
STATIC_WALKABLE = [[is_walkable(cell_to_position((x, y))) for y in range(64)] for x in range(36)]
# A* tile cost as (ground, air or jumping): water 50/7, outside lanes 8, lanes 5.
TILE_COST = [[{'W': (50, 7), '.': (8, 8)}.get(contents[63 - y][x], (5, 5)) for y in range(64)]
             for x in range(36)]


def _path_cache(battle):
    """Results that stay valid until battle.building_cache is rebuilt.

    calculate_building_cache always assigns a new list, so comparing the list's
    identity tells us when buildings were added or destroyed.
    """
    saved = getattr(battle, '_path_cache', None)
    if saved is None or saved[0] is not battle.building_cache:
        saved = battle._path_cache = (battle.building_cache, {})
    return saved[1]


class EntityPathfinder:
    def __init__(self, entity, target, battle_state):
        self.start_position = Position(entity.position.x, entity.position.y)
        self.target_position = Position(target.position.x, target.position.y)
        self.target = target
        self.entity = entity
        self.start_cell = position_to_cell(self.start_position)
        self.battle = battle_state
        self.goals = set()
        self.goal = None

    def heuristic(self, cell):
        x, y = cell
        gx, gy = self.goal
        return 10 * max(abs(x - gx), abs(y - gy))

    def calculate(self):
        cache = _path_cache(self.battle)
        data = self.entity.data
        mover_radius = data.collision_radius
        flies = 1 if (data.is_air_unit or data.jump_speed) else 0
        radius = self.target.data.collision_radius + data.range

        # The first step is to calculate some viable cells that is in attack position.
        # They depend only on these inputs (and the building layout), so reuse them.
        goals_key = ('goals', self.target_position.x, self.target_position.y, radius, mover_radius)
        self.goals = cache.get(goals_key)
        if self.goals is None:
            self.goals = cache[goals_key] = set()
            target_cell = position_to_cell(self.target_position)
            scan_radius = math.ceil(radius*2) + 1
            for x in range(target_cell[0]-scan_radius, target_cell[0]+scan_radius):
                for y in range(target_cell[1]-scan_radius, target_cell[1]+scan_radius):
                    distance = cell_to_position((x, y)).distance_to(self.target_position)
                    # I added 0.375 to radius so that short-ranged troops like lumberjack can reach the tower instead of leering to the side
                    if distance < radius+0.375 and self.battle.pathfind_ground_walkable(cell_to_position((x, y)), mover_radius):
                        self.goals.add((x, y))
        # The second step is to filter goals, only keep the closest one.
        # Not cached: it depends on the exact start position, not just the start cell.
        self.goal = min(self.goals, key=lambda c: cell_to_position(c).distance_to(self.target_position)+cell_to_position(c).distance_to(self.start_position))

        # The A* search depends only on start cell, goal cell, mover size and flying.
        path_key = ('path', self.start_cell, self.goal, mover_radius, flies)
        if path_key not in cache:
            cache[path_key] = self._search(mover_radius, flies)
        return list(cache[path_key])

    def _search(self, mover_radius, flies):
        """A* from start_cell to goal over the half-tile grid."""
        building_cache = self.battle.building_cache
        gx, gy = self.goal
        g = {}
        f = {}
        parent = {}
        closed_set = set()
        g[self.start_cell] = 0
        f[self.start_cell] = self.heuristic(self.start_cell)
        open_heap = [(f[self.start_cell], self.start_cell)]

        while open_heap:
            current_f, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            if current_f > f[current]:
                continue
            if current == self.goal:
                break
            closed_set.add(current)
            px, py = current
            g_current = g[current]
            for neighbor in get_neighboring_points(px, py):
                if neighbor in closed_set: continue
                nx, ny = neighbor
                # Same test as battle.pathfind_ground_walkable(cell_to_position(neighbor), mover_radius).
                if not (STATIC_WALKABLE[nx][ny] and building_cache[nx][ny] > mover_radius):
                    continue
                geo_cost = 14 if nx != px and ny != py else 10
                tentative_g = g_current + TILE_COST[nx][ny][flies] * geo_cost
                if neighbor not in g or tentative_g < g[neighbor]:
                    g[neighbor] = tentative_g
                    parent[neighbor] = current
                    f[neighbor] = tentative_g + 10 * max(abs(nx - gx), abs(ny - gy))  # heuristic
                    heapq.heappush(open_heap, (f[neighbor], neighbor))
        path = [current]
        while path[-1] != self.start_cell:
            path.append(parent[path[-1]])
        path.reverse()

        positions = [cell_to_position(each) for each in path]
        return positions

if __name__ == '__main__':
    from battle import BattleState
    from player import PlayerState

    player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
    player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']
    battle = BattleState(PlayerState(0, player_0_deck, 10), PlayerState(1, player_1_deck, 10))
    battle.deploy_card(0, 'Knight', Position(10.5, 10.5))

    pathfind = EntityPathfinder(battle.entities[7], battle.entities[2], battle)
    print(pathfind.calculate())






