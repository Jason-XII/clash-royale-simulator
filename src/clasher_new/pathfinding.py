from pathlib import Path
import math
from core import Position

grid_path = Path(__file__).with_name('tilemap_lane_grid.txt')
with grid_path.open('r') as f:
    contents = [list(each) for each in f.read().splitlines()]


def position_to_cell(position: Position):
    x, y = position.x, position.y
    return math.floor(2*x), math.floor(2*y)

def cell_to_position(cell):
    x, y = cell
    return Position((x+0.5)/2, (y+0.5)/2)

def heuristic(current, goal):
    return 10 * max(abs(current.x - goal.x), abs(current.y - goal.y))

def get_neighboring_points(x, y):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            new_x, new_y = x+dx, y+dy
            if dx == dy == 0: continue
            if new_x < 0 or new_y < 0 or new_x >= 36 or new_y >= 64: continue
            yield new_x, new_y


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
        return min(
            10 * max(abs(x - gx), abs(y - gy))
            for gx, gy in self.goals
        )

    def calculate(self):
        self.goals = set()
        lane_id = '2' if self.start_cell[0] >= 19 else '1'

        radius = self.target.data.collision_radius + self.entity.data.range - 0.5
        minimum_angle = (15/360)*2*math.pi
        tx, ty = self.target_position.x, self.target_position.y
        for i in range(24):
            dx, dy = radius*math.cos(i*minimum_angle), radius*math.sin(i*minimum_angle)
            new_position = Position(tx+dx, ty+dy)
            if self.battle.ground_walkable(new_position, self.entity.data.collision_radius):
                self.goals.add(position_to_cell(new_position))

        g = {}
        f = {}
        parent = {}
        open_set = {self.start_cell}
        closed_set = set()
        g[self.start_cell] = 0
        f[self.start_cell] = self.heuristic(self.start_cell)

        while open_set:
            current = min(open_set, key=lambda node: f[node])
            if current in self.goals:
                print('Reached goal.')
                break
            open_set.remove(current)
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
                    tile_cost = 800 if not self.entity.data.is_air_unit else 20
                elif tile_char == '.':
                    tile_cost = 20
                elif tile_char != lane_id:
                    tile_cost = 5
                else:
                    tile_cost = 1
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
                    open_set.add(neighbor)
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
    battle.deploy_card(0, 'Knight', Position(9.5, 0.5))

    pathfind = EntityPathfinder(battle.entities[7], battle.entities[2], battle)
    pathfind.calculate()






