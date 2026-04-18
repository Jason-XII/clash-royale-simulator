from arena import TileGrid
from player import PlayerState
from card_mechanics import *
from card_utils import Card, TimedExplosiveData
from fastcore.all import store_attr
import math
from itertools import combinations


class Entity:
    def __init__(self, id, position, player, card_name, battle_state=None):
        store_attr()
        self.data = Card(self.card_name)
        self.is_alive = True
        self.attack_cooldown = self.data.load_time
        self.speed = self.data.speed
        self.battle_state = None
        self.hp = self.data.hp
        self.entity_holder = BasicCharacter(self)
        self.jumping_across_river = False
        # This affects both speed and hit speed.
        self.speed_buff = 1.0
        self.speed_debuff = 1.0
        self.buff_time_remaining = 0.0
        self.debuff_time_remaining = 0.0
        if self.card_name in globals() and not isinstance(self, Projectile):
            self.entity_holder = eval(f"{self.card_name}(self)")
        self.target_id = None
        self.battle_state = battle_state
        self.entity_holder.on_spawn()

    def to_dict(self):
        return {
            'type': 'entity',
            'id': self.id,
            'card_name': self.card_name,
            'player': self.player,
            'x': self.position.x,
            'y': self.position.y,
            'hp': self.hp,
            'max_hp': self.data.hp,
            'attack_cooldown': self.attack_cooldown,
            'speed': self.speed,
            'target_id': self.target_id,
            'jumping_across_river': self.jumping_across_river,
            'deploy_delay_remaining': getattr(self, 'deploy_delay_remaining', 0),
            'collision_radius': self.data.collision_radius if not isinstance(self, Projectile) else 0.3
        }

    def update(self, dt):
        self.entity_holder.on_tick(dt)
        if self.buff_time_remaining > 0:
            self.buff_time_remaining -= dt
        else:
            self.speed_buff = 1.0
        if self.debuff_time_remaining > 0:
            self.debuff_time_remaining -= dt
        else:
            self.speed_debuff = 1.0

    def take_damage(self, amount: float):
        """Apply damage to entity"""
        self.hp -= amount
        if self.hp <= 0 and self.is_alive:
            self.is_alive = False
            self.entity_holder.on_death()
            if self.data.death_damage:
                for entity in self.battle_state.entities.values():
                    if not entity.is_alive or entity.player == self.player: continue
                    if entity.position.distance_to(self.position) - entity.data.collision_radius < 1:
                        entity.take_damage(self.data.death_damage)

    def get_nearest_target(self):
        """Find nearest valid target with priority rules"""
        building_targets = []
        troop_targets = []
        for entity in list(self.battle_state.entities.values()):
            if type(entity).__name__ in {'Projectile', 'SpawnProjectile', 'RollingProjectile', 'AreaEffect', 'TimedExplosive'} or \
                    not (entity.is_alive and entity.player != self.player): continue # exclude spells or stealth entities
            distance = self.position.distance_to(entity.position)
            if (entity.data.is_air_unit and not self.data.attack_air) or ((not entity.data.is_air_unit) and not self.data.attack_ground):
                continue
            if distance-entity.data.collision_radius-self.data.collision_radius <= self.data.sight_range:
                if isinstance(entity, Building):
                    building_targets.append((distance, entity))
                elif not self.data.target_only_buildings:
                    troop_targets.append((distance, entity))

        if self.data.target_only_buildings:
            targets = building_targets
        else:
            targets = troop_targets if troop_targets else building_targets

        targets.sort(key=lambda x: x[0])
        if not targets: return None
        else: return targets[0][1]

    def _should_switch_target(self, current_target, new_target):
        """Determine if we should switch from current target to new target"""
        # if self.position.distance_to(new_target.position)-current_target.data.collision_radius < self.data.sight_range: return False
        if self.data.target_only_buildings and not isinstance(new_target, Building): return False
        if not new_target:
            return True
        if self.position.distance_to(current_target.position) <= self.data.range + current_target.data.collision_radius + self.data.collision_radius:
            return False
        # Always switch to troops in sight range (higher priority than buildings)
        is_current_building = isinstance(current_target, Building)
        is_new_troop = not isinstance(new_target, Building)
        if is_new_troop and is_current_building:
            return True
        if self.position.distance_to(current_target.position) > self.position.distance_to(new_target.position):
            return True
        return False

    def update_current_target(self):
        current_target = None
        if self.target_id is None or \
                self.target_id not in self.battle_state.entities or \
                not self.battle_state.entities.get(self.target_id).is_alive:
            # doesn't have a valid prior target
            self.target_id = None
        else:
            current_target = self.battle_state.entities.get(self.target_id)
            if current_target.position.distance_to(self.position) - current_target.data.collision_radius - self.data.collision_radius > self.data.sight_range:
                current_target = None
                self.target_id = None
        best_target = self.get_nearest_target()
        if self.target_id:
            if self._should_switch_target(self.battle_state.entities[self.target_id], best_target):
                current_target = best_target
                self.target_id = current_target.id if current_target else None
        else:
            current_target = best_target
            self.target_id = current_target.id if current_target else None
        return current_target

    def create_projectile(self, target):
        if not self.data.projectiles: raise Exception('Entity does not have any projectiles.')
        projectile = Projectile(
            id=self.battle_state.next_entity_id, position=Position(self.position.x, self.position.y),
            player=self.player, source_card_name=self.data.name, target=target)
        projectile.battle_state = self.battle_state
        self.battle_state.entities[projectile.id] = projectile
        self.battle_state.next_entity_id += 1

    def on_both_sides_of_river(self, e2):
        if isinstance(e2, Entity):
            y = e2.position.y
        else: y = e2.y
        if y < 15.0: return self.position.y > 17.0
        else: return self.position.y < 15.0

    def near_river(self):
        return abs(self.position.y-15.0)<self.data.collision_radius or abs(self.position.y-17.0)<self.data.collision_radius

class Troop(Entity):
    def __init__(self, id, position, player, card_name, battle_state=None):
        super().__init__(id, position, player, card_name, battle_state)
        self.deploy_delay_remaining = self.data.deploy_time
        self.name = self.data.name
        self.path_blocked_counter = 0
        self.jumping_across_river = False
        self.start_jumping_position = None
        self.spawned = False

    def to_dict(self):
        d = super().to_dict()
        d.update({'type': 'troop', 'path_blocked_counter': self.path_blocked_counter,
                  'start_jumping_position': (self.start_jumping_position.x, self.start_jumping_position.y)
                  if self.start_jumping_position else None})
        return d

    def move_towards(self, position, dt: float) -> None:
        dx, dy = position.x-self.position.x, position.y-self.position.y
        distance = math.hypot(dx, dy)
        if distance == 0: return
        move_distance = min(self.speed * dt * self.speed_buff * self.speed_debuff, distance)
        move_x, move_y = (dx / distance) * move_distance, (dy / distance) * move_distance

        # Check if the new position would be walkable (for ground units)
        new_position = Position(self.position.x + move_x, self.position.y + move_y)

        # Air units ignore walkability checks, ground units must check
        if self.data.is_air_unit or (self.battle_state.ground_walkable(new_position, self.data.collision_radius)):
            self.position.x += move_x
            self.position.y += move_y
            self.path_blocked_counter -= 1 if self.path_blocked_counter else 0
        else:
            # If direct path is blocked, try to find a way around
            self.path_blocked_counter += 1 if self.path_blocked_counter <= 3 else 0
            original_angle = math.atan2(move_y, move_x)
            move_distance = math.hypot(move_x, move_y)
            angle_offsets = [i * math.pi / 16 for i in range(1, 17)] + [-i * math.pi / 16 for i in range(1, 17)]
            for angle_offset in angle_offsets:
                new_angle = original_angle + angle_offset
                new_move_x = math.cos(new_angle) * move_distance
                new_move_y = math.sin(new_angle) * move_distance
                if self.battle_state.ground_walkable(Position(self.position.x+new_move_x, self.position.y+new_move_y),
                                                self.data.collision_radius):
                    # print('Preparing to move to:', self.position.x+new_move_x, self.position.y+new_move_y)
                    if new_move_x*move_x+new_move_y*move_y >= -0.0001:
                        self.position.x += new_move_x
                        self.position.y += new_move_y
                        break
                    else:
                        # print('Failed', new_move_x*move_x+new_move_y*move_y)
                        pass

    def _get_pathfind_target(self, target_entity: 'Entity') -> Position:
        """Get pathfinding target using priority system with advanced post-tower-destruction logic:
        Air units: 1) Targets in FOV, 2) Towers (fly directly over river)
        Ground units:
        - Before first tower destroyed: 1) Troops in sight range, 2) Bridge center, 3) Princess towers
        - After first tower destroyed: 1) Troops in FOV, 2) Center bridge, 3) Cross bridge if clear, 4) Target buildings
        """
        final_target = target_entity.position
        need_to_cross = (self.position.y - 16.0) * (final_target.y - 16.0) < 0
        if self.data.is_air_unit or not need_to_cross:
            return final_target
        return self._get_basic_pathfind_target()

    def _get_basic_pathfind_target(self) -> Position:
        """If no target in sight, where should the troop walk to? """
        near_left =  abs(self.position.x - 3.5) < abs(self.position.x - 14.5)
        on_bridge = (abs(self.position.x - (3.5 if near_left else 14.5)) <= 1.5 and
                    abs(self.position.y - 16.0) <= 1.0)
        before_bridge = (self.position.y < 15.0 and self.player==0) or (self.position.y > 17.0 and self.player==1)
        if self.data.is_air_unit:
            if self.player == 0:
                return TileGrid.RED_LEFT_TOWER if near_left else TileGrid.RED_RIGHT_TOWER
            else:
                return TileGrid.BLUE_LEFT_TOWER if near_left else TileGrid.BLUE_RIGHT_TOWER
        if before_bridge and not on_bridge:
            possible_x = [3, 14]
            possible_y = [15, 17]
            return Position(min(possible_x, key=lambda x: abs(self.position.x - x)),
                            min(possible_y, key=lambda y: abs(self.position.y - y)))
        else:
            if self.player == 0:
                if near_left: target = TileGrid.RED_LEFT_TOWER
                else: target = TileGrid.RED_RIGHT_TOWER
                if self.battle_state.ground_walkable(target, self.data.collision_radius) and self.position.y >= 25.0:
                    target = TileGrid.RED_KING_TOWER
                return target
            else:
                if near_left: target = TileGrid.BLUE_LEFT_TOWER
                else: target = TileGrid.BLUE_RIGHT_TOWER
                if self.battle_state.ground_walkable(target, self.data.collision_radius) and self.position.y <= 7.0:
                    target = TileGrid.BLUE_KING_TOWER
                return target

    def update(self, dt):
        if not self.is_alive: return
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return # Haven't finished deploying yet
        # Logic: the troop may have a current target (or doesn't), and `get_nearest_target` also gives a
        # recommended target. If current target exists, compare that with the recommendation to see
        # if it needs to switch. If it doesn't exist, use the best target. However, the best target may also
        # be none.
        super().update(dt)
        if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position):
            self.jumping_across_river = False
            self.data.is_air_unit = Card(self.name).is_air_unit
            self.speed = self.data.speed
        current_target = self.update_current_target()

        if current_target:
            # Move towards target if out of attack range
            distance = self.position.distance_to(current_target.position)
            if distance > (self.data.range + current_target.data.collision_radius + self.data.collision_radius) or self.jumping_across_river:
                has_jump_ability = self.data.jump_speed and self.on_both_sides_of_river(current_target) and self.near_river()
                if self.data.is_air_unit or has_jump_ability:
                    pathfind_target = current_target.position
                else:
                    pathfind_target = self._get_pathfind_target(current_target)
                if not self.jumping_across_river and has_jump_ability:
                    self.start_jumping_position = Position(self.position.x, self.position.y)
                    self.jumping_across_river = True
                    self.data.is_air_unit = True
                    self.speed = self.data.jump_speed
                self.move_towards(pathfind_target, dt)
                self.attack_cooldown = max(self.data.hit_speed-self.data.load_time, self.attack_cooldown-dt*self.speed_buff*self.speed_debuff)
            else:
                if self.attack_cooldown <= 0:
                    self.entity_holder.on_attack(current_target)
                else:
                    self.attack_cooldown -= dt*self.speed_buff*self.speed_debuff
        else:
            # now calculate:
            if self.data.is_air_unit:
                self.move_towards(self._get_basic_pathfind_target(), dt)
                return
            near_left = abs(self.position.x - 3.5) < abs(self.position.x - 14.5)
            before_bridge = (self.position.y < 15.0 and self.player == 0) or (
                        self.position.y > 17.0 and self.player == 1)
            if near_left and before_bridge and self.player == 0:
                if 2.5 <= self.position.x <= 4.5:
                    dx = 0
                elif self.position.x < 2.5:
                    dx = 2.5-self.position.x
                else:
                    dx = 4.5-self.position.x
                dy = 15.0-self.position.y
            elif not near_left and before_bridge and self.player == 0:
                if 13.5 <= self.position.x <= 15.5:
                    dx = 0
                elif self.position.x < 13.5:
                    dx = 13.5-self.position.x
                else:
                    dx = 15.5-self.position.x
                dy = 15.0-self.position.y
            elif self.player == 0:
                self.move_towards(self._get_basic_pathfind_target(), dt)
                return
            elif near_left and before_bridge and self.player == 1:
                if 2.5 <= self.position.x <= 4.5:
                    dx = 0
                elif self.position.x < 2.5:
                    dx = 2.5-self.position.x
                else:
                    dx = 4.5-self.position.x
                dy = 17.0-self.position.y
            elif not near_left and before_bridge and self.player == 1:
                if 13.5 <= self.position.x <= 15.5:
                    dx = 0
                elif self.position.x < 13.5:
                    dx = 13.5 - self.position.x
                else:
                    dx = 15.5 - self.position.x
                dy = 17.0 - self.position.y
            else:
                self.move_towards(self._get_basic_pathfind_target(), dt)
                return
            distance = dt * self.data.speed
            dx = (dx/math.hypot(dx, dy))*distance
            dy = (dy/math.hypot(dx, dy))*distance
            real_dx = (1 if dx > 0 else -1 if dx < 0 else 0) * distance / math.sqrt(2)
            real_dy = (1 if dy > 0 else -1 if dy < 0 else 0) * distance / math.sqrt(2)
            if not self.path_blocked_counter:
                self.move_towards(Position(self.position.x + real_dx, self.position.y + real_dy), dt)
            else:
                self.move_towards(Position(self.position.x + dx, self.position.y + dy), dt)


class Building(Entity):
    def __init__(self, id, position, player, card_name, persistent=False):
        super().__init__(id, position, player, card_name)
        self.deploy_delay_remaining = self.data.deploy_time
        self.lifetime_elapsed = 0.0
        self.target_id = None
        self.tower_active = False
        self.persistent = persistent
        self.name = self.data.name

    def to_dict(self):
        d = super().to_dict()
        d.update({'type': 'building', 'lifetime_elapsed': self.lifetime_elapsed,
                  'tower_active': self.tower_active, 'persistent': self.persistent})
        return d

    def take_damage(self, amount: float):
        super().take_damage(amount)
        if self.data.name == 'KingTower' and not self.tower_active:
            self.tower_active = True

    def update(self, dt: float):
        """Update building - only attack, no movement"""
        if not self.is_alive: return
        if self.data.name == 'KingTower' and not self.tower_active: return
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return
        if self.data.lifetime > 0 and not self.persistent:
            decay = (self.data.hp / float(self.data.lifetime)) * dt
            self.take_damage(decay)
        if self.attack_cooldown > 0:
            self.attack_cooldown = max(0, self.attack_cooldown-dt*self.speed_buff*self.speed_debuff)
        target = self.update_current_target()
        if target and self.attack_cooldown <= 0:
            if self.data.projectiles:
                self.create_projectile(target)
            else:
                target.take_damage(self.data.damage)
            self.attack_cooldown = self.data.hit_speed

class Projectile(Entity):
    def __init__(self, id, position, player, source_card_name, target, homing=True):
        super().__init__(id, position, player, source_card_name)
        self.target_position = Position(target.position.x, target.position.y)
        self.proj = self.data.projectile_data # a shortcut
        self.homing = homing
        self.target = target
        self.battle_state = None
        self.name = self.proj.name
        self.data.collision_radius = 0.3

    def to_dict(self):
        d = super().to_dict()
        d.update({'type': 'projectile', 'homing': self.homing,
                  'target_position': (self.target_position.x, self.target_position.y)
                  if self.target_position else None})
        return d

    def update(self, dt):
        """Update projectile - move towards target"""
        if not self.is_alive: return
        target_position_final = self.target_position if not self.homing else self.target.position
        distance = self.position.distance_to(target_position_final)
        if distance <= self.proj.speed * dt:
            if not self.proj.radius:
                self.target.take_damage(self.proj.damage)
                if self.proj.buff_time:
                    self.target.speed_debuff = min(1 + self.proj.target_buff['speedMultiplier'] / 100, self.target.speed_debuff)
                    self.target.debuff_time_remaining = self.proj.buff_time
            else:
                self._deal_splash_damage()
            # Now handle target buff

            self.is_alive = False
        else:
            self._move_towards(target_position_final, dt)

    def _deal_splash_damage(self) -> None:
        """Deal damage to entities in splash radius using hitbox overlap detection"""
        for entity in list(self.battle_state.entities.values()):
            if entity.player == self.player or not entity.is_alive: continue
            if entity.data.is_air_unit and not self.proj.hits_air: continue
            if (not entity.data.is_air_unit) and not self.proj.hits_ground: continue

            # Use hitbox-based collision detection for more accurate splash damage
            if entity.position.distance_to(self.target_position) <= (self.proj.radius + entity.data.collision_radius):
                entity.take_damage(self.proj.damage)
                if self.proj.buff_time:
                    entity.speed_debuff = min(1 + self.proj.target_buff['speedMultiplier'] / 100, entity.speed_debuff)
                    entity.debuff_time_remaining = self.proj.buff_time

    def _move_towards(self, target_pos, dt):
        """Move towards target position"""
        # Note: I used a much cleaner way of writing the code.
        direction = complex(target_pos.x - self.position.x, target_pos.y - self.position.y)
        step = direction / abs(direction) * self.proj.speed * dt
        self.position.x += step.real
        self.position.y += step.imag


class TimedExplosive(Entity):
    def __init__(self, id, position, player, card_name):
        super().__init__(id, position, player, card_name)
        self.dsd = TimedExplosiveData(self.data.death_spawn_data)
        self.deploy_delay_remaining = self.dsd.deploy_time
        self.name = self.dsd.name

    def update(self, dt):
        if not self.is_alive: return
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return
        for entity in self.battle_state.entities.values():
            if not entity.is_alive or entity.player == self.player: continue
            if entity.position.distance_to(self.position) - entity.data.collision_radius < self.dsd.range:
                if entity.name in ('King_PrincessTowers', 'KingTower'):
                    entity.take_damage(self.dsd.damage*self.dsd.crown_tower_damage_percent)
                else:
                    entity.take_damage(self.dsd.damage)
        self.is_alive = False

    def take_damage(self, amount: float):
        # Bombs does not take damage!
        pass


def get_spawn_position(card_info, position, player, offset_angle=True):
    spawn_number, spawn_delay, r = card_info.spawn_number, card_info.spawn_delay, card_info.spawn_radius
    if spawn_number == 1: return [Position(position.x, position.y)]
    positions = []
    angle_offset = {2: 0, 3: math.pi/2, 4: math.pi/4, 6: 0}
    for i in range(spawn_number):
        angle = 2*math.pi*i/spawn_number
        if offset_angle: angle += angle_offset.get(spawn_number, 0)
        if player == 1: angle += math.pi
        dx, dy = r*math.cos(angle), r*math.sin(angle)
        positions.append(Position(position.x+dx, position.y+dy))
    return positions


class BattleState:
    def __init__(self, player_0: PlayerState, player_1: PlayerState):
        self.entities = {}
        self.players = [player_0, player_1]
        self.arena = TileGrid()
        self.time = 0.0
        self.tick = 0
        self.dt = 1 / 60
        self.game_over = False
        self.winner = None
        self.next_entity_id = 1
        self.regen = 2.8

        self._spawn_entity(Building(1, self.arena.RED_LEFT_TOWER, 1, 'King_PrincessTowers', True))
        self._spawn_entity(Building(2, self.arena.RED_RIGHT_TOWER, 1, 'King_PrincessTowers', True))
        self._spawn_entity(Building(3, self.arena.BLUE_LEFT_TOWER, 0, 'King_PrincessTowers', True))
        self._spawn_entity(Building(4, self.arena.BLUE_RIGHT_TOWER, 0, 'King_PrincessTowers', True))
        self._spawn_entity(Building(5, self.arena.RED_KING_TOWER, 1, 'KingTower', True))
        self._spawn_entity(Building(6, self.arena.BLUE_KING_TOWER, 0, 'KingTower', True))

        self.schedule = []

    def in_river(self, position):
        river_tiles = [(0, 15), (0, 16), (1, 15), (1, 16),
            *[(i, j) for i in range(5, 13) for j in range(15, 17)], # (5, 15) to (12, 16)
            (16, 15), (16, 16), (17, 15), (17, 16)]
        return (int(position.x), int(position.y)) in river_tiles

    def ensure_walkability(self, entity):
        if entity.jumping_across_river and self.in_river(entity.position): return
        if isinstance(entity, Building) or isinstance(entity, Projectile): return

        if not self.ground_walkable(entity.position, entity.data.collision_radius):

            x, y, r = entity.position.x, entity.position.y, entity.data.collision_radius
            push_ratio = 0.5
            if y < push_ratio*r: y=push_ratio*r
            elif y > 32-push_ratio*r: y=32-push_ratio*r
            if x < push_ratio*r: x=r
            elif x > 18-push_ratio*r: x=18-push_ratio*r
            if 15-push_ratio*r < y < 17+push_ratio*r and not entity.data.is_air_unit:
                y = 15-push_ratio*r if y-15 < 17-y else 17+push_ratio*r
            entity.position.x = x
            entity.position.y = y

    def _spawn_entity(self, entity):
        self.ensure_walkability(entity)
        entity.battle_state = self
        entity.id = self.next_entity_id
        self.entities[len(self.entities)+1] = entity
        self.next_entity_id += 1

    def delayed_spawn(self, entity, delay):
        if delay:
            self.schedule.append((entity, self.time+delay))
        else:
            self._spawn_entity(Troop(*entity))

    def update_player_hp(self):
        p0, p1 = self.players
        p0.king_tower_hp = self.entities[6].hp
        p0.left_tower_hp = self.entities[3].hp
        p0.right_tower_hp = self.entities[4].hp
        p1.king_tower_hp = self.entities[5].hp
        p1.left_tower_hp = self.entities[1].hp
        p1.right_tower_hp = self.entities[2].hp

    def step(self, dt):
        if self.game_over: return
        self.update_player_hp()
        p0 = self.players[0].get_crown_count()
        p1 = self.players[1].get_crown_count()
        p0h = self.players[0]
        p1h = self.players[1]
        if p0 == 3:
            self.game_over = True
            self.winner = 1
            return
        elif p1 == 3:
            self.game_over = True
            self.winner = 0
            return
        elif 300>self.time >= 180:
            if p0 > p1:
                self.game_over = True
                self.winner = 1
                return
            elif p0 < p1:
                self.game_over = True
                self.winner = 0
                return
        elif self.time >= 300:
            self.game_over = True
            min_0_hp = min(each for each in (p0h.king_tower_hp, p0h.left_tower_hp, p0h.right_tower_hp) if each > 0)
            min_1_hp = min(each for each in (p1h.king_tower_hp, p1h.left_tower_hp, p1h.right_tower_hp) if each > 0)
            if min_0_hp > min_1_hp:
                self.winner = 0
            else:
                self.winner = 1
        for each in self.players:
            each.regenerate_elixir(dt, 2.8 if self.time < 120 else 1.4 if self.time < 240 else 2.8/3)
        for entity in list(self.entities.values()):
            entity.update(dt)
            self.ensure_walkability(entity)
        self.resolve_collisions()

        for entity, spawn_time in self.schedule:
            if self.time > spawn_time: self._spawn_entity(Troop(*entity))
        self.schedule = [each for each in self.schedule if each[1] > self.time]
        self.time += dt
        self.tick += 1

    def deploy_card(self, player_id, card_name, position):
        if not self.players[player_id].can_play_card(card_name):
            return False
        card_info = Card(card_name)

        positions = get_spawn_position(card_info, position, player_id)
        delayed_counter = card_info.spawn_delay
        for p in positions:
            self.delayed_spawn((len(self.entities)+1, p, player_id, card_name, self), delayed_counter)
            delayed_counter += card_info.spawn_delay
        self.players[player_id].play_card(card_name)
        return True

    def ground_walkable(self, position, mover_radius):
        if not self.arena.is_walkable(position): return False
        return not self.is_position_occupied_by_building(position, mover_radius)

    def is_position_occupied_by_building(self, position, mover_radius: float = 0.5) -> bool:
        """Return True when a position overlaps any live building footprint."""
        for entity in self.entities.values():
            if not isinstance(entity, Building) or not entity.is_alive:
                continue
            if position.distance_to(entity.position) < (entity.data.collision_radius + mover_radius):
                return True
        return False

    def resolve_collisions(self):
        entities_alive = [each for each in self.entities.values() if each.is_alive and (isinstance(each, Troop) or isinstance(each, Building))]
        ground_troops = combinations([each for each in entities_alive if not each.data.is_air_unit], 2)
        flying_troops = combinations([each for each in entities_alive if each.data.is_air_unit], 2)
        for troop in (ground_troops, flying_troops):
            for e1, e2 in troop:
                if e1.position.distance_to(e2.position) < e1.data.collision_radius + e2.data.collision_radius:
                    overlap = e1.data.collision_radius + e2.data.collision_radius - e1.position.distance_to(e2.position)
                    # the direction vector points from e1 to e2
                    direction_vector = complex(e2.position.x-e1.position.x, e2.position.y-e1.position.y)
                    if abs(direction_vector) == 0: return
                    direction_vector /= abs(direction_vector)
                    movement_ratio = e2.data.speed / (e1.data.speed+e2.data.speed)
                    e2.position.x += direction_vector.real*movement_ratio*overlap
                    e2.position.y += direction_vector.imag*movement_ratio*overlap
                    e1.position.x += -direction_vector.real * (1-movement_ratio)*overlap
                    e1.position.y += -direction_vector.imag * (1-movement_ratio)*overlap

    def on_death(self, entity):
        if entity.name == 'King_PrincessTowers':
            player = entity.player
            for each in self.entities.values():
                if each.name == 'KingTower' and each.player == player:
                    each.tower_active = True
                    break

    def deal_area_damage(self, from_player, position, range, amount, attack_air, attack_ground):
        for entity in self.entities.values():
            if not entity.is_alive or entity.player == from_player: continue
            if attack_air and entity.data.is_air_unit:
                if entity.position.distance_to(position) < range:
                    entity.take_damage(amount)
            elif attack_ground and not entity.data.is_air_unit:
                if entity.position.distance_to(position) < range:
                    entity.take_damage(amount)


