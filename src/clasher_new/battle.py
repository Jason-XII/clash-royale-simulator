from .arena import TileGrid, Position
from .player import PlayerState
from .card_utils import Card
from fastcore.all import store_attr
import math

class Entity:
    def __init__(self, id, position, player, card_name):
        store_attr()
        self.data = Card(self.card_name)
        self.is_alive = True
        self.attack_cooldown = 0
        self.speed = self.data.speed
        self.battle_state = None

    def update(self, dt): raise NotImplementedError

    def take_damage(self, amount: float):
        """Apply damage to entity"""
        self.data.hp -= amount
        if self.data.hp <= 0 and self.is_alive:
            self.is_alive = False

    def get_nearest_target(self):
        """Find nearest valid target with priority rules"""
        building_targets = []
        troop_targets = []
        for entity in list(self.battle_state.entities.values()):
            if type(entity).__name__ in {'Projectile', 'SpawnProjectile', 'RollingProjectile', 'AreaEffect'} or \
                    not (entity.is_alive and entity.player != self.player): continue # exclude spells or stealth entities
            distance = self.position.distance_to(entity.position)
            if (entity.data.is_air_unit and not self.data.attack_air) or ((not entity.data.is_air_unit) and not self.data.attack_ground):
                continue
            if distance <= self.data.sight_range:
                if isinstance(entity, Building):
                    building_targets.append((distance, entity))
                elif not self.data.target_only_buildings:
                    troop_targets.append((distance, entity))

        if self.data.target_only_buildings:
            targets = building_targets
        else:
            targets = troop_targets if troop_targets else building_targets

        targets.sort()
        if not targets: return None
        else: return targets[0][1]

    def _should_switch_target(self, current_target, new_target):
        """Determine if we should switch from current target to new target"""
        if self.position.distance_to(new_target.position) < self.data.sight_range: return False
        if self.data.target_only_buildings and not isinstance(new_target, Building): return False
        if self.position.distance_to(current_target.position) <= self.data.range + current_target.data.collision_radius:
            return False
        # Always switch to troops in sight range (higher priority than buildings)
        is_current_building = isinstance(current_target, Building)
        is_new_troop = not isinstance(new_target, Building)
        if is_new_troop and is_current_building:
            return True
        return False

class Troop(Entity):
    def __init__(self, id, position, player, card_name):
        super().__init__(position, player, card_name)
        self.deploy_delay_remaining = self.data.deploy_time
        self.target_id = None

    def move_towards(self, position, dt: float, battle_state=None) -> None:
        dx, dy = position.x-self.position.x, position.y-self.position.y
        distance = math.hypot(dx, dy)
        move_distance = min(self.speed * dt, distance)
        move_x, move_y = (dx / distance) * move_distance, (dy / distance) * move_distance

        # Check if the new position would be walkable (for ground units)
        new_position = Position(self.position.x + move_x, self.position.y + move_y)

        # Air units ignore walkability checks, ground units must check
        if self.data.is_air_unit or (self.battle_state.ground_walkable(new_position, self.data.collision_radius)):
            self.position.x += move_x
            self.position.y += move_y
        else:
            # If direct path is blocked, try to find a way around
            original_angle = math.atan2(move_y, move_x)
            move_distance = math.hypot(move_x, move_y)
            angle_offsets = [i * math.pi / 8 for i in range(1, 9)] + [-i * math.pi / 8 for i in range(1, 9)]
            new_move_x, new_move_y = None, None
            for angle_offset in angle_offsets:
                new_angle = original_angle + angle_offset
                new_move_x = math.cos(new_angle) * move_distance
                new_move_y = math.sin(new_angle) * move_distance
                if battle_state.ground_walkable(Position(self.position.x+new_move_x, self.position.y+new_move_y),
                                                self.data.collision_radius):
                    break
            self.position.x += new_move_x
            self.position.y += new_move_y

    def update(self, dt):
        if not self.is_alive: return
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return # Haven't finished deploying yet
        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt
        if self.target_id is None or \
            self.target_id not in self.battle_state.entities or \
                not self.battle_state.entities.get(self.target_id).is_alive:
            self.target_id = None
        best_target = self.get_nearest_target()
        if (best_target and not self.target_id) or \
                self._should_switch_target(self.battle_state.entities[self.target_id], best_target):
            current_target = best_target
            self.target_id = current_target.id

class BattleState:
    def __init__(self, player_0: PlayerState, player_1: PlayerState):
        self.entities = {}
        self.players = [player_0, player_1]
        self.arena = TileGrid()
        self.time = 0.0
        self.tick = 0
        self.dt = 1 / 30  # 33ms per tick (~30 FPS)
        self.game_over = False
        self.winner = None
        self.next_entity_id = 1
        self.regen = 2.8

        self._spawn_entity(Building(1, self.arena.RED_LEFT_TOWER, 1, 'King_PrincessTowers', []))
        self._spawn_entity(Building(2, self.arena.RED_RIGHT_TOWER, 1, 'King_PrincessTowers', []))
        self._spawn_entity(Building(3, self.arena.BLUE_LEFT_TOWER, 0, 'King_PrincessTowers', []))
        self._spawn_entity(Building(4, self.arena.BLUE_RIGHT_TOWER, 0, 'King_PrincessTowers', []))
        self._spawn_entity(Building(5, self.arena.RED_KING_TOWER, 1, 'KingTower', []))
        self._spawn_entity(Building(6, self.arena.BLUE_KING_TOWER, 0, 'KingTower', []))

    def _spawn_entity(self, entity):
        entity.battle_state = self
        self.entities[len(self.entities)+1] = entity
        entity.on_spawn()
        self.next_entity_id += 1

    def step(self, dt):
        for each in self.players:
            each.regenerate_elixir(dt, 2.8 if self.time < 120 else 1.4 if self.time < 240 else 2.8/3)
        for entity in list(self.entities.values()):
            entity.update(dt, self)
        self.time += dt
        self.tick += 1

    def deploy_card(self, player_id, card_name, position):
        if not self.players[player_id].can_play_card(card_name):
            return False
        self._spawn_entity(Troop(len(self.entities)+1, position, player_id, card_name, []))
        return True

    def ground_walkable(self, position, mover_radius):
        if not self.arena.is_walkable(position): return False
        return not self.is_position_occupied_by_building(position, mover_radius)

    def is_position_occupied_by_building(self, position, mover_radius: float = 0.5) -> bool:
        """Return True when a position overlaps any live building footprint."""
        for entity in self.entities.values():
            if not isinstance(entity, Building) or not entity.is_alive:
                continue
            if position.distance_to(entity.position) < (entity.data.collision_radius + mover_radius) * 0.95:
                return True
        return False


