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


class EntityPathfinder:
    def __init__(self, entity, target, battle_state):
        self.start_position = Position(entity.position.x, entity.position.y)
        self.target_position = Position(target.position.x, target.position.y)
        self.target = target
        self.entity = entity
        self.start_cell = position_to_cell(self.start_position)
        self.battle = battle_state
        self.goals = set()

    def heuristic(self, cell):
        x, y = cell
        gx, gy = self.goal
        return 10 * max(abs(x - gx), abs(y - gy))

    def calculate(self):
        self.goals = set()

        radius = self.target.data.collision_radius + self.entity.data.range
        # The first step is to calculate some viable cells that is in attack position.

        target_cell = position_to_cell(self.target_position)
        scan_radius = math.ceil(radius*2) + 1
        for x in range(target_cell[0]-scan_radius, target_cell[0]+scan_radius):
            for y in range(target_cell[1]-scan_radius, target_cell[1]+scan_radius):
                distance = cell_to_position((x, y)).distance_to(self.target_position)
                # I added 0.375 to radius so that short-ranged troops like lumberjack can reach the tower instead of leering to the side
                if distance < radius+0.375 and self.battle.ground_walkable(cell_to_position((x, y)), self.entity.data.collision_radius):
                    self.goals.add((x, y))
        # The second step is to filter goals, only keep the closest one.
        self.goal = min(self.goals, key=lambda c: cell_to_position(c).distance_to(self.start_position))

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
            for neighbor in get_neighboring_points(current[0], current[1]):
                if neighbor in closed_set: continue
                neighbor_position = cell_to_position(neighbor)
                if not self.battle.ground_walkable(neighbor_position, self.entity.data.collision_radius):
                    continue
                nx, ny = neighbor
                px, py = current
                tile_char = contents[63-ny][nx]
                if tile_char == 'W':
                    tile_cost = 800 if not self.entity.data.is_air_unit else 7
                elif tile_char == '.':
                    tile_cost = 8
                else:
                    tile_cost = 5
                if nx != px and ny != py:
                    geo_cost = 14
                else:
                    geo_cost = 10
                step_cost = tile_cost * geo_cost
                tentative_g = g[current] + step_cost
                if neighbor not in g or tentative_g < g[neighbor]:
                    g[neighbor] = tentative_g
                    parent[neighbor] = current
                    f[neighbor] = g[neighbor] + self.heuristic((nx, ny))
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






