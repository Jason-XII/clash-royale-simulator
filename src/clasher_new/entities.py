import random

from .card_utils import Card
import math
from .arena import Position, TileGrid


class Entity:
    def __init__(self, id, position, player, card_name, mechanics):
        self.id, self.position, self.player, self.card_name, self.mechanics = \
            id, position, player, card_name, mechanics
        self.data = Card(self.card_name)
        self.tower_active = False # This attribute only works with KingTowers
        self.is_alive = True
        self.stun_timer = 0.0
        self.slow_timer = 0.0
        self.slow_multiplier = 1.0
        self.attack_speed_buff = 1.0
        self.attack_speed_debuff = 1.0
        self.attack_cooldown = 0
        self.speed = self.data.speed
        self.battle_state = None
        for each in self.mechanics:
            each.on_attach(self)

    def update(self, dt: float, battle_state): raise NotImplementedError
    def take_damage(self, amount: float):
        """Apply damage to entity"""
        print(self.card_name, amount)
        for mechanic in self.mechanics:
            guard = getattr(mechanic, "take_damage_during_dash", None)
            if callable(guard):
                should_take_damage = guard(self, amount)
                if not should_take_damage:
                    return

        if self.data.name == "KingTower" and amount > 0:
            self.tower_active = True
        self.data.hp -= amount
        if self.data.hp <= 0 and self.is_alive:
            self.is_alive = False
            self.on_death()  # Trigger death mechanics

    def on_spawn(self) -> None:
        """Some characters may spawn with effects like electric wizards and ice wizards"""
        for mechanic in self.mechanics:
            mechanic.on_spawn(self)

    def on_death(self) -> None:
        """Some characters have special effects on death like the Golem spawns two small Golems on death"""
        for mechanic in self.mechanics:
            mechanic.on_death(self)
    
    def _deal_attack_damage(self, primary_target, damage: float, battle_state) -> None:
        """Deal damage to target, with splash damage if applicable"""
        area_damage_radius = 0
        if not primary_target.is_alive: return
        if self.data.area_damage_radius is not None:
            area_damage_radius = self.data.area_damage_radius / 1000.0
        elif self.data.projectile_damage_radius is not None:
            area_damage_radius = self.data.projectile_damage_radius / 1000.0

        primary_target.take_damage(damage)

        if area_damage_radius > 0:
            for entity in list(battle_state.entities.values()):
                if entity == primary_target or entity.player_id == self.player: continue
                entity_distance = primary_target.position.distance_to(entity.position)
                entity_radius = self.data.collision_radius/1000.0 or 0.5
                if entity_distance <= (area_damage_radius + entity_radius):
                    entity.take_damage(damage)

    def apply_stun(self, duration: float):
        """Apply stun effect for specified duration"""
        if duration > self.stun_timer: self.stun_timer = duration
        # Stun resets/restarts attack timing for units already in the attack loop.
        attack_interval = (self.data.hit_speed / 1000.0) / (self.attack_speed_debuff * self.attack_speed_buff)
        self.attack_cooldown = max(self.attack_cooldown, attack_interval)
        # Allow card mechanics to react to stun (e.g., Sparky charge reset).
        for mechanic in getattr(self, "mechanics", []):
            handler = getattr(mechanic, "handle_stun", None)
            if callable(handler):
                handler(self)
        
    def apply_slow(self, duration: float, multiplier: float) -> None:
        """Apply slow effect for specified duration"""
        self.slow_timer = max(self.slow_timer, duration)
        self.slow_multiplier = min(self.slow_multiplier, multiplier)
        self.attack_speed_debuff = min(self.attack_speed_debuff, multiplier)
        self.speed = self.data.speed * self.slow_multiplier
    
    def update_status_effects(self, dt: float) -> None:
        """Update status effect timers"""
        if self.stun_timer > 0: self.stun_timer -= dt
        if self.slow_timer > 0:
            self.slow_timer -= dt
            if self.slow_timer <= 0:
                self.speed = self.data.speed
                self.slow_multiplier = 1.0
                self.attack_speed_debuff = 1.0

    def _is_valid_target(self, entity: 'Entity') -> bool:
        """Check if entity can be targeted (excludes spell entities)"""
        # Spell entities cannot be targeted by troops
        spell_entity_types = {'Projectile', 'SpawnProjectile', 'RollingProjectile', 'AreaEffect'}
        if type(entity).__name__ in spell_entity_types: return False
        stealth_until = getattr(entity, '_stealth_until', 0)
        if stealth_until and hasattr(entity, 'battle_state'):
            current_ms = int(entity.battle_state.time * 1000)
            if stealth_until > current_ms:
                return False
        return entity.is_alive and entity.player != self.player
    
    def can_attack_target(self, target: 'Entity') -> bool:
        """Check if this entity can attack the target"""
        if not self._is_valid_target(target): return False
        if ((target.data.is_air_unit and not self.data.attack_air) or
                ((not target.data.is_air_unit) and not self.data.attack_ground)): return False
        distance = self.position.distance_to(target.position)
        return distance <= self.data.range + target.data.collision_radius

    def get_nearest_target(self, entities):
        """Find nearest valid target with priority rules"""
        building_targets = []
        troop_targets = []
        for entity in entities.values():
            if not self._is_valid_target(entity): continue # exclude spells or stealth entities
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
    
    def _should_switch_target(self, current_target: 'Entity', new_target: 'Entity') -> bool:
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
    def __init__(self, id, position, player, card_name, mechanics):
        super().__init__(id, position, player, card_name, mechanics)
        self.is_charging  = False
        self.charge_target_position = None
        self.distance_traveled = 0.0
        self.initial_position = None
        self.deploy_delay_remaining = self.data.deploy_time
        self.initial_position = Position(self.position.x, self.position.y)
        self.target_id = None

    def _update_charging_state(self) -> None:
        """Update charging state - check if troop should start charging based on distance traveled"""
        distance_traveled = self.position.distance_to(self.initial_position)
        charge_distance_tiles = self.data.charge_range / 1000.0
        if distance_traveled >= charge_distance_tiles and not self.is_charging:
            self.is_charging = True
            self.speed *= 2


    def move_towards(self, position, dt: float, battle_state=None) -> None:
        dx, dy = position.x-self.position.x, position.y-self.position.y
        distance = math.hypot(dx, dy)
        move_distance = min(self.speed * dt, distance)
        move_x, move_y = (dx / distance) * move_distance, (dy / distance) * move_distance

        # Check if the new position would be walkable (for ground units)
        new_position = Position(self.position.x + move_x, self.position.y + move_y)

        # Air units ignore walkability checks, ground units must check
        if self.data.is_air_unit or (battle_state.ground_walkable(new_position, self.data.collision_radius)):
            self.position.x += move_x
            self.position.y += move_y
        else:
            # If direct path is blocked, try to find a way around
            alternative_move = self._find_alternative_move(move_x, move_y, battle_state)
            if alternative_move:
                alt_x, alt_y = alternative_move
                self.position.x += alt_x
                self.position.y += alt_y

    def _find_alternative_move(self, original_move_x, original_move_y, battle_state):
        """Find an alternative movement direction when the direct path is blocked."""
        if not battle_state: return None
        original_angle = math.atan2(original_move_y, original_move_x)
        move_distance = math.hypot(original_move_x, original_move_y)
        angle_offsets = [i * math.pi/8 for i in range(1, 9)] + [-i * math.pi/8 for i in range(1, 9)]
        for angle_offset in angle_offsets:
            new_angle = original_angle + angle_offset
            new_move_x = math.cos(new_angle) * move_distance
            new_move_y = math.sin(new_angle) * move_distance
            alt_position = Position(self.position.x + new_move_x, self.position.y + new_move_y)

            if battle_state.ground_walkable(alt_position, self.data.collision_radius):
                return new_move_x, new_move_y
        return None

    def _create_projectile(self, target: 'Entity', battle_state: 'BattleState') -> None:
        """Create a projectile towards the target"""
        if self.data.name == "Bowler":
            dx, dy = target.position.x - self.position.x, target.position.y - self.position.y
            distance = math.hypot(dx, dy)
            direction_x = dx / distance
            direction_y = dy / distance
            rolling_projectile = RollingProjectile(
                id=battle_state.next_entity_id,
                position=Position(self.position.x, self.position.y),
                player_id=self.player,
                card_stats=self.data,
                hitpoints=1,
                max_hitpoints=1,
                damage=self.data.projectile_data.damage, # bowler doesn't charge
                range=self.data.projectile_data.radius / 1000.0,  # Use splash radius as rolling width
                sight_range=0,
                travel_speed=self.data.projectile_data.speed,  # Convert back to tiles/min for RollingProjectile
                projectile_range=7.5,  # 7500 game units = 7.5 tiles
                spawn_delay=0.0,  # No spawn delay for Bowler
                spawn_character=None,  # Bowler doesn't spawn units
                spawn_character_data=None,
                knockback_distance=self.data.projectile_data.pushback / 1000.0,
                target_direction_x=direction_x,
                target_direction_y=direction_y
            )

            battle_state.entities[rolling_projectile.id] = rolling_projectile
        else:
            # Create regular projectile
            projectile = Projectile(
                id=battle_state.next_entity_id,
                position=Position(self.position.x, self.position.y),
                player=self.player, source_card_name=self.data.name,
                target_position=Position(target.position.x, target.position.y),
            )

            battle_state.entities[projectile.id] = projectile

        battle_state.next_entity_id += 1

    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update troop - move and attack"""

        if not self.is_alive: return
        if self.stun_timer > 0:
            self.stun_timer = max(0.0, self.stun_timer-dt)
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return # Haven't finished deploying yet
        self.update_status_effects(dt) # for now, it's only slow and stun
        for mechanic in self.mechanics:
            mechanic.on_tick(self, dt * 1000)  # Convert to ms
        
        # Update attack cooldown and track time for visualization
        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt / (self.attack_speed_buff * self.attack_speed_debuff)
        if self.data.charge_range: self._update_charging_state()


        current_target = None
        if self.target_id is not None:
            current_target = battle_state.entities.get(self.target_id)
            if not current_target or not current_target.is_alive: self.target_id = None
        
        # Always check for better targets (troops in FOV take priority over buildings)
        best_target = self.get_nearest_target(battle_state.entities)
        if best_target and (not current_target or self._should_switch_target(current_target, best_target)):
            current_target = best_target
            self.target_id = current_target.id
        
        if current_target:
            # Move towards target if out of range
            distance = self.position.distance_to(current_target.position)
            if distance > (self.data.range + current_target.data.collision_radius):
                if self.data.is_air_unit:
                    pathfind_target = current_target.position
                else:
                    pathfind_target = self._get_pathfind_target(current_target, battle_state)
                self.move_towards(pathfind_target, dt, battle_state)
            elif self.attack_cooldown <= 0:
                for mechanic in self.mechanics:
                    mechanic.on_attack_start(self, current_target)
                # Check if this troop uses projectiles
                if self.data.projectiles:
                    self._create_projectile(current_target, battle_state)
                else:
                    # Direct attack with special charging damage if applicable
                    if self.is_charging: attack_damage = self.data.charge_damage
                    else: attack_damage = self.data.damage
                    self._deal_attack_damage(current_target, attack_damage, battle_state)

                for mechanic in self.mechanics:
                    mechanic.on_attack_hit(self, current_target)
                self.attack_cooldown = 1 / (self.data.hit_speed/1000)
                if self.data.charge_range and self.is_charging:
                    self.is_charging = False
                    self.charge_target_position = None
                    self.speed /= 2
        else:
            self.move_towards(self._get_basic_pathfind_target(), dt, self.battle_state)


    def _get_pathfind_target(self, target_entity: 'Entity', battle_state=None) -> Position:
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
        """Original pathfinding logic before first tower is destroyed"""
        near_left =  abs(self.position.x - 3.5) < abs(self.position.x - 14.5)
        on_bridge = (abs(self.position.x - (3.5 if near_left else 14.5)) <= 1.5 and
                    abs(self.position.y - 16.0) <= 1.0)
        before_bridge = (self.position.y < 16.0 and self.player==0) or (self.position.y > 16.0 and self.player==1)
        if before_bridge and not on_bridge:
            possible_x = [3, 14]
            possible_y = [15, 17]
            return Position(min(possible_x, key=lambda x: abs(self.position.x - x)),
                            min(possible_y, key=lambda y: abs(self.position.y - y)))
        else:
            if self.player == 0:
                return TileGrid.RED_LEFT_TOWER if near_left else TileGrid.RED_RIGHT_TOWER
            else:
                return TileGrid.BLUE_LEFT_TOWER if near_left else TileGrid.BLUE_RIGHT_TOWER
  

class Building(Entity):
    def __init__(self, id, position, player, card_name, mechanics):
        super().__init__(id, position, player, card_name, mechanics)
        self.deploy_delay_remaining = self.data.deploy_time
        self.lifetime_elapsed = 0.0
        self.target_id = None
    
    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update building - only attack, no movement"""
        if not self.is_alive or self.stun_timer > 0: return
        if self.data.name == 'KingTower' and not self.tower_active: return
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining = max(0.0, self.deploy_delay_remaining - dt)
            return
        self.update_status_effects(dt)
        for mechanic in self.mechanics:
            mechanic.on_tick(self, dt * 1000)
        if self.data.lifetime > 0:
            decay = (self.data.hp / float(self.data.lifetime)) * dt
            if decay > 0: self.take_damage(decay)
        if self.attack_cooldown > 0: self.attack_cooldown -= dt * (self.attack_speed_debuff * self.attack_speed_buff)
        target = self.get_nearest_target(battle_state.entities)
        self.target_id = target.id if target else None
        if target and self.can_attack_target(target) and self.attack_cooldown <= 0:
            for mechanic in self.mechanics:
                mechanic.on_attack_start(self, target)
            if self.data.projectiles:
                self._create_projectile(target, battle_state)
            else:
                self._deal_attack_damage(target, self.data.damage, battle_state)
            for mechanic in self.mechanics:
                mechanic.on_attack_hit(self, target)
            self.attack_cooldown = 1 / (self.data.hit_speed*self.attack_speed_buff*self.attack_speed_debuff/1000)
    
    def _create_projectile(self, target: 'Entity', battle_state: 'BattleState') -> None:
        """Create a projectile towards the target"""
        projectile = Projectile(
            id=battle_state.next_entity_id, position=Position(self.position.x, self.position.y),
            player=self.player, source_card_name=self.data.name,
            target_position=Position(target.position.x, target.position.y),
        )
        battle_state.entities[projectile.id] = projectile
        battle_state.next_entity_id += 1


class Projectile(Entity):
    def __init__(self, id, position, player, source_card_name, target_position):
        # No mechanics field for projectile
        super().__init__(id, position, player, source_card_name, [])
        print('Projectile created')
        self.target_position = target_position
        self.proj = self.data.projectile_data # a shortcut
    
    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update projectile - move towards target"""
        if not self.is_alive: return
        distance = self.position.distance_to(self.target_position)
        if distance <= self.proj.speed * dt:
            self._deal_splash_damage(battle_state)
            self.is_alive = False
        else:
            self._move_towards(self.target_position, dt)
    
    def _move_towards(self, target_pos: Position, dt: float) -> None:
        """Move towards target position"""
        # Note: I used a much cleaner way of writing the code.
        direction = complex(target_pos.x - self.position.x, target_pos.y - self.position.y)
        step = direction / abs(direction) * self.proj.speed * dt
        self.position.x += step.real
        self.position.y += step.imag
    
    def _deal_splash_damage(self, battle_state: 'BattleState') -> None:
        """Deal damage to entities in splash radius using hitbox overlap detection"""
        for entity in list(battle_state.entities.values()):
            if entity.player == self.player or not entity.is_alive: continue
            if entity.data.is_air_unit and not self.proj.hits_air: continue
            if (not entity.data.is_air_unit) and not self.proj.hits_ground: continue
            
            # Use hitbox-based collision detection for more accurate splash damage
            if entity.position.distance_to(self.target_position) <= (self.proj.radius + entity.data.collision_radius):
                entity.take_damage(self.proj.damage)
                if slow:=self.proj.target_buff.get('speedMultiplier'):
                    entity.apply_slow(self.proj.buff_time, slow)
                if self.proj.pushback > 0 and not isinstance(entity, Building):
                    self._apply_knockback(entity, battle_state)

    def _apply_knockback(self, entity: 'Entity', battle_state: 'BattleState') -> None:
        """Push entity away from impact center."""
        dx = entity.position.x - self.target_position.x
        dy = entity.position.y - self.target_position.y
        distance = math.hypot(dx, dy)
        if distance == 0: return
        new_position = Position(
            entity.position.x + (dx / distance) * self.proj.pushback,
            entity.position.y + (dy / distance) * self.proj.pushback,
        )
        # This might be buggy. What if the position is not walkable?
        if battle_state.is_ground_position_walkable(new_position, entity):
            entity.position = new_position


class AreaEffect(Entity):
    """Area effect spells that stay on the ground for a duration"""
    def __init__(self, id, position, player, card_name):
        super().__init__(id, position, player, card_name, [])
        self.time_alive = 0.0
        self.effect = self.data.area_effect_data
        self.pull_force = 1

    
    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update area effect - apply effects and check duration"""
        if not self.is_alive: return
        self.time_alive += dt

        if self.time_alive >= self.effect.duration:
            self.is_alive = False
            return

        for entity in list(battle_state.entities.values()):
            if entity.player == self.player or not entity.is_alive or entity == self: continue
            if self.data.is_air_unit and not self.data.attack_air: continue
            if (not self.data.is_air_unit) and not self.data.attack_ground: continue
            if self._hitbox_overlaps_with_radius(entity):
                distance = entity.position.distance_to(self.position)
                if self.card_name == 'Tornado':
                    self._apply_tornado_pull(entity, distance, dt, battle_state)
                if self.card_name == 'Freeze':
                    entity.apply_stun(max(dt, 0.1))
                    entity.apply_slow(max(dt, 0.1), 0.0)
                else:
                    entity.apply_slow(max(dt, 0.25), self.effect.speed_multiplier)
                
                # Apply damage over time (small damage each tick)
                if self.effect.damage > 0:
                    damage = self.effect.damage * dt
                    if isinstance(entity, Building):
                        damage *= 3.5 if self.data.name == 'Earthquake' else 1.0
                        if self.data.name in {"Tower", "KingTower"}:
                            damage *= self.effect.crown_tower_damage_percent
                    entity.take_damage(damage)

    def _apply_tornado_pull(self, entity: 'Entity', distance: float, dt: float, battle_state: 'BattleState') -> None:
        """Pull entity towards tornado center"""
        if distance == 0:
            return
        dx = self.position.x - entity.position.x
        dy = self.position.y - entity.position.y
        pull_distance = self.pull_force * dt

        if pull_distance > distance:
            pull_distance = distance * 0.9  # Stop just short of center
        pull_x = (dx / distance) * pull_distance
        pull_y = (dy / distance) * pull_distance
        new_position = Position(entity.position.x + pull_x, entity.position.y + pull_y)
        if entity.data.is_air_unit or battle_state.is_ground_position_walkable(new_position, entity):
            entity.position.x += pull_x
            entity.position.y += pull_y
    
    def _hitbox_overlaps_with_radius(self, entity: 'Entity') -> bool:
        """Check if entity's hitbox overlaps with area effect radius"""
        distance = entity.position.distance_to(self.position)
        return distance <= (self.data.collision_radius + entity.data.collision_radius)


class SpawnProjectile(Projectile):
    """Projectile that spawns units when it reaches target"""
    def __init__(self, id, position, player, target_position, card_name='GoblinBarrel'):
        super().__init__(id, position, player, card_name, target_position)
        self.spawn_count = 3
        self.spawn_character = "Goblin"
        self.spawn_character_data = None
        self.activation_delay = 0.0
        self.time_alive  = 0.0
    
    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update projectile - move towards target and spawn units on impact"""
        if not self.is_alive: return
        self.time_alive += dt
        if self.time_alive < self.activation_delay: return
        distance = self.position.distance_to(self.target_position)
        if distance <= self.proj.speed * dt:
            self._spawn_units(battle_state)
            self.is_alive = False
        else:
            self._move_towards(self.target_position, dt)
    
    def _spawn_units(self, battle_state: 'BattleState') -> None:
        """Spawn units at target position"""
        spawn_radius = 1.0  # tiles
        for _ in range(self.spawn_count):
            angle = random.random() * 2 * math.pi
            distance = random.random() * spawn_radius
            spawn_x = self.target_position.x + distance * math.cos(angle)
            spawn_y = self.target_position.y + distance * math.sin(angle)
            battle_state._spawn_troop(Position(spawn_x, spawn_y), self.player)

