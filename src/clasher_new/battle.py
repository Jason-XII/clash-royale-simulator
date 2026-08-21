from core import BlankEntity
from player import PlayerState
from pathfinding_heap import EntityPathfinder
from card_mechanics import *
from card_utils import Card, TimedExplosiveData, spells, buildings
import math
from itertools import combinations


class Entity:
    def __init__(self, id, position, player, card_name, battle_state: "BattleState" = None):
        # Stores permanent information about this entity like `player` and `card_name`.
        self.id, self.position, self.player, self.card_name, self.battle_state = (id, position, player, card_name, battle_state)
        self.data = Card(self.card_name)
        self.name = self.data.name

        # Stores state information that is likely to change.
        self.is_alive = True
        self.attack_cooldown = self.data.hit_speed-self.data.load_time
        self.speed = self.data.speed
        self.hp = self.data.hp
        self.shield_health = self.data.shield_health
        self.target_id = None

        # Why use both targetable and invincible? Because some entities like the royal ghost/archer queen can be invisible but
        # still takes damage. Other entities like the bandit/boss bandit/golden knight/miner(underground) can not be hit in a
        # certain state.
        self.targetable = True
        self.invincible = False

        # There are a lot of entities that can jump across the arena's river, and the movement pattern is
        # significantly different, so I dedicated a special variable to store this information.
        self.jumping_across_river = False

        # This affects both speed and hit speed. I will rewrite this when poison comes out.
        self.speed_buff = 1.0
        self.speed_debuff = 1.0
        self.buff_time_remaining = 0.0
        self.debuff_time_remaining = 0.0

        # This part is where flexibility comes in - some cards have special mechanics that can't be handled in
        # the entity/troop/buildings classes. So I created `BasicCharacter` to delegate most of the logic.
        # If a card doesn't have special logic like the knight and mini-pekka, then only `BasicCharacter` will be
        # used.
        self.entity_holder = BasicCharacter(self)
        if self.card_name in globals() and not isinstance(self, Projectile):
            self.entity_holder = eval(f"{self.card_name}(self)")
        self.entity_holder.on_spawn()

        self.path = []

        self.pending_damage = []

    def to_dict(self):
        """If I want to render a certain entity on the screen, what's the minimal information I'll need?"""
        return {
            'type': 'entity',
            'card_name': self.card_name,
            'player': self.player,
            'x': self.position.x,
            'y': self.position.y,
            'hp': self.hp,
            'max_hp': self.data.hp,
            'shield_max_hp': self.data.shield_health,
            'shield_hp': self.shield_health,
            'collision_radius': self.data.collision_radius if not isinstance(self, Projectile) else 0.3
        }

    def die(self):
        """Automatically call entity holder's on_death to prevent bugs"""
        self.is_alive = False
        self.entity_holder.on_death()
        self.battle_state.on_death(self)

    def update(self, dt):
        # This part may be a bit confusing because it doesn't check the `is_alive` and `deploy_delay_remaining` attribute.
        # Reasons: this will be eventually called by `super()` and won't terminate the actual update function. And
        # there are miner and drill that needs to be moving before it's even deployed. So this function only updates the buff_time
        # and debuff_time attribute.

        # I assume this function will be called after the deployment and alive check.
        self.entity_holder.on_tick(dt)
        if self.buff_time_remaining > 0:
            self.buff_time_remaining -= dt
        else:
            self.speed_buff = 1.0
        if self.debuff_time_remaining > 0:
            self.debuff_time_remaining -= dt
        else:
            self.speed_debuff = 1.0

        for pending_damage in self.pending_damage:
            self.take_damage(pending_damage, delayed=False)
        self.pending_damage = []


    def take_damage(self, amount: float, delayed=False):
        """Apply damage to entity"""
        if self.invincible: return
        if delayed:
            self.pending_damage.append(amount)
            return
        if not self.shield_health: self.hp -= amount
        else: self.shield_health = max(0, self.shield_health - amount)

        if self.hp <= 0 and self.is_alive:
            self.die()
            if self.data.death_damage:
                # I assume that all death damage deals attack to both air and ground troops.
                # The game data file hasn't specified what's the radius of the death damage,
                # so here I just set it to 1 tile
                self.battle_state.deal_area_damage(self.player, self.position, 1.0+self.data.collision_radius, self.data.death_damage,
                                                   attack_air=True, attack_ground=True)

    def in_attack_range(self, target):
        if 'PrincessTower' in target.name:
            bonus = 0.5
        else:
            bonus = 0
        return self.position.distance_to(target.position) <= self.data.range + target.data.collision_radius + bonus
    def in_sight_range(self, target):
        if 'PrincessTower' in target.name:
            bonus = 0.5
        else:
            bonus = 0
        return self.position.distance_to(target.position) <= self.data.sight_range + target.data.collision_radius + bonus

    def get_nearest_target(self):
        """Find nearest valid target with priority rules"""
        building_targets = []
        troop_targets = []
        for entity in list(self.battle_state.entities.values()):
            if not isinstance(entity, Troop) and not isinstance(entity, Building): continue
            if not entity.is_alive or entity.player == self.player: continue
            if not entity.targetable: continue
            distance = self.position.distance_to(entity.position)
            if (entity.data.is_air_unit and not self.data.attack_air) or ((not entity.data.is_air_unit) and not self.data.attack_ground):
                continue
            if self.in_sight_range(entity):
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
        # If target is killed or no longer in sight, update the target_id to None
        current_target = None
        if self.target_id is None or \
                self.target_id not in self.battle_state.entities or \
                not self.battle_state.entities.get(self.target_id).is_alive:
            # doesn't have a valid prior target
            self.target_id = None
            self.path = []
        else:
            current_target = self.battle_state.entities.get(self.target_id)
            if not self.in_sight_range(current_target):
                if 'PrincessTower' not in current_target.name:
                    self.path = []
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

        # Now, the current target can still be None (example: a knight deployed at the back)
        # This case we update the target to the nearest enemy princess tower, so we can do A* globally!
        if self.target_id is None:
            min_distance = float('inf')
            self.target_id = 1
            for i in range(1, 7):
                if not self.battle_state.entities[i].is_alive: continue
                possible_princess_tower = self.battle_state.entities[i]
                if possible_princess_tower.player == self.player: continue
                distance = possible_princess_tower.position.distance_to(self.position) - possible_princess_tower.data.collision_radius
                if distance < min_distance:
                    min_distance = distance
                    self.target_id = i
            current_target = self.battle_state.entities[self.target_id]
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
        d.update({'type': 'troop', })
        return d

    def move_towards(self, position, dt: float, can_overshoot=False) -> None:
        dx, dy = position.x-self.position.x, position.y-self.position.y
        distance = math.hypot(dx, dy)
        if distance == 0: return
        if not can_overshoot:
            move_distance = min(self.speed * dt * self.speed_buff * self.speed_debuff, distance)
        else:
            move_distance = self.speed * dt * self.speed_buff * self.speed_debuff
        move_x, move_y = (dx / distance) * move_distance, (dy / distance) * move_distance
        self.position.x += move_x
        self.position.y += move_y

    def update(self, dt):
        if not self.is_alive: return
        if self.name == 'Miner':
            super().update(dt)
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return # Haven't finished deploying yet
        # Logic: the troop may have a current target (or doesn't), and `get_nearest_target` also gives a
        # recommended target. If current target exists, compare that with the recommendation to see
        # if it needs to switch. If it doesn't exist, use the best target. However, the best target may also
        # be none.
        if self.name != 'Miner':
            super().update(dt)
        # The miner needs to update before deployment.
        if self.jumping_across_river and self.on_both_sides_of_river(self.start_jumping_position):
            self.jumping_across_river = False
            self.data.is_air_unit = Card(self.name).is_air_unit
            self.speed = self.data.speed
        current_target = self.update_current_target()
        # After the modification, we always have a target, sometimes it's in sight range, sometimes it's not
        # We use A* search for all cases to pathfind towards the target.
        # The case is even the same with ground troops and air troops.

        # Move towards target if out of attack range
        if (not self.in_attack_range(current_target)) or self.jumping_across_river:
            has_jump_ability = self.data.jump_speed and self.on_both_sides_of_river(current_target) and self.near_river() and self.in_sight_range(current_target)
            if not self.jumping_across_river and has_jump_ability:
                self.start_jumping_position = Position(self.position.x, self.position.y)
                self.jumping_across_river = True
                self.data.is_air_unit = True
                self.speed = self.data.jump_speed
            if self.data.is_air_unit:
                self.move_towards(current_target.position, dt, True)
            else:
                if not self.path:
                    self.path = EntityPathfinder(self, current_target, self.battle_state).calculate()
                elif self.in_sight_range(current_target) and self.battle_state.tick % 3 == 0:
                    self.path = EntityPathfinder(self, current_target, self.battle_state).calculate()

                # determine the next waypoint and move towards that waypoint
                min_point = min(self.path, key=lambda pos: pos.distance_to(self.position))
                index = self.path.index(min_point)
                start_vector = (self.position.x-self.path[0].x, self.position.y-self.path[0].y)
                close_vector = (self.position.x-min_point.x, self.position.y-min_point.y)
                dot = start_vector[0]*close_vector[0] + start_vector[1]*close_vector[1]
                if dot >= 0:
                    # move towards next waypoint
                    index += 1
                if index == len(self.path):
                    self.move_towards(current_target.position, dt, True)
                else:
                    self.move_towards(self.path[index], dt, True)
            self.attack_cooldown = max(self.data.hit_speed-self.data.load_time, self.attack_cooldown-dt*self.speed_buff*self.speed_debuff)
        else:
            if self.attack_cooldown <= 0:
                self.entity_holder.on_attack(current_target)
            else:
                self.attack_cooldown -= dt*self.speed_buff*self.speed_debuff



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
        d.update({'type': 'building'})
        return d

    def take_damage(self, amount: float, delayed=False):
        super().take_damage(amount, delayed)
        if self.data.name == 'KingTower' and not self.tower_active:
            self.tower_active = True

    def update(self, dt: float):
        """Update building - only attack, no movement"""
        if not self.is_alive: return
        if self.data.name == 'KingTower' and not self.tower_active: return
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return
        super().update(dt)
        if self.data.lifetime > 0 and not self.persistent:
            decay = (self.data.hp / float(self.data.lifetime)) * dt
            self.take_damage(decay)
            self.take_damage(decay)
        if self.attack_cooldown > 0:
            self.attack_cooldown = max(0, self.attack_cooldown-dt*self.speed_buff*self.speed_debuff)
        target = self.update_current_target()
        if target and self.in_attack_range(target) and self.attack_cooldown <= 0:
            if self.data.projectiles:
                self.create_projectile(target)
            else:
                target.take_damage(self.data.damage)
            self.attack_cooldown = self.data.hit_speed

class Projectile(Entity):
    def __init__(self, id, position, player, source_card_name, target, homing=True, battle_state=None):
        super().__init__(id, position, player, source_card_name)
        self.target_position = Position(target.position.x, target.position.y)
        self.initial_position = Position(self.position.x, self.position.y)
        self.proj = self.data.projectile_data # a shortcut
        self.rolling = bool(self.proj.roll_range)
        self.homing = homing
        self.target = target
        self.battle_state = battle_state
        self.name = self.proj.name
        if self.data.type == 'spell':
            self.data.collision_radius = self.proj.radius
        else: self.data.collision_radius = 0.3

        self.damage_dealt = []

    def to_dict(self):
        d = super().to_dict()
        d.update({'type': 'projectile'})
        return d

    def update(self, dt):
        """Update projectile - move towards target"""
        if not self.is_alive: return
        if self.rolling:
            distance = self.position.distance_to(self.initial_position)
            if distance > self.proj.roll_range:
                self.is_alive = False
                return
            # now deal area damage
            for each in self.battle_state.entities.values():
                if type(each).__name__ in {'Projectile', 'SpawnProjectile', 'RollingProjectile', 'AreaEffect',
                                              'TimedExplosive'}: continue  # exclude spells or stealth entities
                if each in self.damage_dealt or each.data.is_air_unit: continue
                if not each.is_alive or each.player == self.player: continue
                if each.position.distance_to(self.position) < each.data.collision_radius + self.proj.radius:
                    each.take_damage(self.proj.damage, delayed=True)
                    self.damage_dealt.append(each)
                    # now knockback
                    direction_vector = complex(each.position.x-self.position.x, each.position.y-self.position.y)
                    direction_vector /= abs(direction_vector)
                    direction_vector *= self.proj.pushback
                    if isinstance(each, Troop):
                        new_x = each.position.x + direction_vector.real
                        new_y = each.position.y + direction_vector.imag
                        if self.battle_state.ground_walkable(Position(new_x, new_y), each.data.collision_radius):
                            each.position = Position(new_x, new_y)
            direction_vector = complex(self.target_position.x-self.initial_position.x,
                                       self.target_position.y-self.initial_position.y)
            direction_vector /= abs(direction_vector)
            direction_vector *= self.proj.speed * dt
            self.position.x += direction_vector.real
            self.position.y += direction_vector.imag
            return

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
            if entity.invincible: continue
            if entity.player == self.player or not entity.is_alive: continue
            if entity.data.is_air_unit and not self.proj.hits_air: continue
            if (not entity.data.is_air_unit) and not self.proj.hits_ground: continue

            # Use hitbox-based collision detection for more accurate splash damage
            if entity.position.distance_to(self.target_position) <= (self.proj.radius + entity.data.collision_radius):
                amount_dealt = self.proj.damage if "King" not in entity.name else round(self.proj.damage * self.proj.crown_tower_percent)
                entity.take_damage(amount_dealt)
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
        self.building_positions = []

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
        self.entities[self.next_entity_id] = entity
        self.next_entity_id += 1

    def _wrap(self, entity_data):
        card_name = entity_data[3]
        entity_data = list(entity_data)
        entity_data[0] = self.next_entity_id
        self.next_entity_id += 1
        if len(entity_data) == 7:
            return Projectile(*entity_data)
        if card_name in spells:
            return Entity(*entity_data)
        elif card_name in buildings:
            return Building(*entity_data)
        else:
            return Troop(*entity_data)

    def delayed_spawn(self, entity, delay):
        if delay:
            self.schedule.append((entity, self.time+delay))
        else:
            self._spawn_entity(self._wrap(entity))

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
        self.entities = {key:value for key,value in self.entities.items() if (value.is_alive or key <= 6)}
        self.building_positions = [(entity.position.x, entity.position.y, entity.data.collision_radius) for entity in self.entities.values() if isinstance(entity, Building)]
        for entity in list(self.entities.values()):
            entity.update(dt)
            self.ensure_walkability(entity)
        self.resolve_collisions()

        for entity, spawn_time in self.schedule:
            if self.time >= spawn_time: self._spawn_entity(self._wrap(entity))
        self.schedule = [each for each in self.schedule if each[1] > self.time]
        self.time += dt
        self.tick += 1

    def deploy_card(self, player_id, card_name, position):
        if not self.players[player_id].can_play_card(card_name):
            return False
        card_info = Card(card_name)

        if card_info.type != 'spell':
            # Check the deployment area is legit
            if self.is_position_occupied_by_building(position, 0): return False
            if player_id == 0:
                if position.y <= 1.0 and (position.x <= 6.0 or position.x > 12.0): return False
                if position.y >= 21.0: return False
                elif position.y >= 15.0:
                    if position.x <= 9:
                        if self.players[1].left_tower_hp > 0: return False
                    else:
                        if self.players[1].right_tower_hp > 0: return False
            elif player_id == 1:
                if position.y > 31.0 and (position.x <= 6.0 or position.x > 12.0): return False
                if position.y <= 10: return False
                elif position.y <= 17.0:
                    if position.x <= 9:
                        if self.players[0].left_tower_hp > 0: return False
                    else:
                        if self.players[0].right_tower_hp > 0: return False

        if card_info.type == 'spell' and card_info.projectiles:
            initial_position = self.arena.BLUE_KING_TOWER if player_id == 0 else self.arena.RED_KING_TOWER

            target = BlankEntity(position)
            delayed_counter = 0
            for wave in range(card_info.projectile_waves):
                initial_position = Position(initial_position.x, initial_position.y)
                # I know that I should not use `len(self.entities)+1` here because it would cause bugs.
                # so in the actual `delay_spawn` function, I added another layer that corrects the entity id to a legit one.
                self.delayed_spawn((len(self.entities)+1, initial_position, player_id, card_name, target, False, self), delayed_counter)
                delayed_counter += card_info.wave_interval
            self.players[player_id].play_card(card_name)
            return True

        positions = get_spawn_position(card_info, position, player_id)
        delayed_counter = 0
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
        for x,y,r in self.building_positions:
            # I choose not to use math.hypot to speed things up. This functino gets called several millions times per game
            if (x-position.x)**2+ (y-position.y)**2 < (r + mover_radius)**2:
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

    def deal_area_damage(self, from_player, position, range, amount, attack_air, attack_ground, crown_tower_damage_percent=1.0):
        for entity in self.entities.values():
            if not entity.is_alive or entity.player == from_player: continue
            if entity.invincible: continue
            amount_dealt = amount if "King" not in entity.name else amount*crown_tower_damage_percent
            if attack_air and entity.data.is_air_unit:
                if entity.position.distance_to(position) < range:
                    entity.take_damage(amount_dealt)
            elif attack_ground and not entity.data.is_air_unit:
                if entity.position.distance_to(position) < range:
                    entity.take_damage(amount_dealt)


