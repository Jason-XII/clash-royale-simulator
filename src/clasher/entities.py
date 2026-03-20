from card_utils import Card
import math
from .battle import BattleState
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
        for each in self.mechanics:
            each.on_attach(self)

    def update(self, dt: float, battle_state): raise NotImplementedError
    def take_damage(self, amount: float):
        """Apply damage to entity"""
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
        nearest = None
        min_distance = float('inf')
        building_targets = []
        troop_targets = []
        
        for entity in entities.values():
            if not self._is_valid_target(entity): continue # exclude spells or stealth entities
            distance = self.position.distance_to(entity.position)
            if (entity.is_air_unit and not self.data.attack_air) or ((not entity.is_air_unit) and not self.data.attack_ground):
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
        return sorted(targets)[0][1] # returns nearest entity
    
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

    def _move_towards_target(self, target_entity: 'Entity', dt: float, battle_state=None) -> None:
        """Move towards target entity with simple obstacle avoidance"""
        # Air units fly directly to targets
        if self.data.is_air_unit: pathfind_target = target_entity.position
        else: pathfind_target = self._get_pathfind_target(target_entity, battle_state)
        dx, dy = pathfind_target.x-self.position.x, pathfind_target.y-self.position.y
        distance = math.hypot(dx, dy)
        move_distance = max((self.speed / 60.0) * dt, distance)
        move_x, move_y = (dx / distance) * move_distance, (dy / distance) * move_distance

        # Check if the new position would be walkable (for ground units)
        new_position = Position(self.position.x + move_x, self.position.y + move_y)

        # Air units ignore walkability checks, ground units must check
        if self.data.is_air_unit or (battle_state.is_ground_position_walkable(new_position, self)):
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

            if battle_state.is_ground_position_walkable(alt_position, self):
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
        if self.stun_timer > 0: return
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
                self._move_towards_target(current_target, dt, battle_state)
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
        if on_bridge:
            if self.player == 0:
                return TileGrid.RED_LEFT_TOWER if near_left else TileGrid.RED_RIGHT_TOWER
            else:
                return TileGrid.BLUE_LEFT_TOWER if near_left else TileGrid.BLUE_RIGHT_TOWER
        else:
            possible_x = [2, 3, 4, 13, 14, 15]
            return Position(max(possible_x, key=lambda x: abs(self.position.x - x)), 16.0)
  

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
            decay = (self.data.hp / float(self.data.lifetime)) * (dt * 1000.0)
            self.take_damage(decay)
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

    def _hitbox_overlaps_with_splash(self, entity: 'Entity') -> bool:
        """Check if entity's hitbox overlaps with splash damage radius"""
        # Get entity collision radius (default to 0.5 tiles if not specified or None)
        if entity.card_stats and hasattr(entity.card_stats,
                                         'collision_radius') and entity.card_stats.collision_radius is not None:
            entity_radius = entity.card_stats.collision_radius
        else:
            entity_radius = 0.5

        # Calculate distance between projectile impact and entity center
        distance = entity.position.distance_to(self.target_position)

        # Check if splash radius overlaps with entity hitbox
        return distance <= (self.splash_radius + entity_radius)
    
    def _deal_splash_damage(self, battle_state: 'BattleState') -> None:
        """Deal damage to entities in splash radius using hitbox overlap detection"""
        for entity in list(battle_state.entities.values()):
            if entity.player == self.player or not entity.is_alive: continue
            if entity.data.is_air_unit and not self.proj.hits_air: continue
            if (not self.proj.is_air) and not self.proj.hits_ground: continue
            
            # Use hitbox-based collision detection for more accurate splash damage
            if self._hitbox_overlaps_with_splash(entity):
                damage = self.damage
                if isinstance(entity, Building) and getattr(entity.card_stats, 'name', None) in {"Tower", "KingTower"}:
                    damage *= self.crown_tower_damage_multiplier
                entity.take_damage(damage)
                if self.stun_duration > 0:
                    entity.apply_stun(self.stun_duration)
                if self.slow_duration > 0 and self.slow_multiplier < 1.0:
                    entity.apply_slow(self.slow_duration, self.slow_multiplier)
                if self.knockback_distance > 0 and not isinstance(entity, Building):
                    self._apply_knockback(entity, battle_state)

    def _apply_knockback(self, entity: 'Entity', battle_state: 'BattleState') -> None:
        """Push entity away from impact center."""
        dx = entity.position.x - self.target_position.x
        dy = entity.position.y - self.target_position.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            return
        new_position = Position(
            entity.position.x + (dx / distance) * self.knockback_distance,
            entity.position.y + (dy / distance) * self.knockback_distance,
        )
        if getattr(entity, "is_air_unit", False) or battle_state.is_ground_position_walkable(new_position, entity):
            entity.position = new_position

@dataclass
class AreaEffect(Entity):
    """Area effect spells that stay on the ground for a duration"""
    duration: float = 4.0
    freeze_effect: bool = False
    speed_multiplier: float = 1.0
    radius: float = 3.0
    time_alive: float = 0.0
    hits_air: bool = True
    hits_ground: bool = True
    crown_tower_damage_multiplier: float = 1.0
    building_damage_multiplier: float = 1.0
    
    # Tornado-specific properties
    pull_force: float = 0.0
    is_tornado: bool = False
    
    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update area effect - apply effects and check duration"""
        if not self.is_alive:
            return
        
        self.time_alive += dt
        
        # Check if duration expired
        if self.time_alive >= self.duration:
            self.is_alive = False
            return
        
        # Apply effects to entities in radius
        for entity in list(battle_state.entities.values()):
            if entity.player_id == self.player_id or not entity.is_alive or entity == self:
                continue

            is_air = getattr(entity, 'is_air_unit', False)
            if is_air and not self.hits_air:
                continue
            if (not is_air) and not self.hits_ground:
                continue
            
            # Use hitbox-based collision detection  
            if self._hitbox_overlaps_with_radius(entity):
                distance = entity.position.distance_to(self.position)
                
                # Apply tornado pull effect
                if self.is_tornado and self.pull_force > 0:
                    self._apply_tornado_pull(entity, distance, dt, battle_state)
                
                if self.freeze_effect:
                    entity.apply_stun(max(dt, 0.1))
                    entity.apply_slow(max(dt, 0.1), 0.0)
                elif self.speed_multiplier < 1.0:
                    entity.apply_slow(max(dt, 0.25), self.speed_multiplier)
                
                # Apply damage over time (small damage each tick)
                if self.damage > 0:
                    damage = self.damage * dt
                    if isinstance(entity, Building):
                        damage *= self.building_damage_multiplier
                        if getattr(entity.card_stats, 'name', None) in {"Tower", "KingTower"}:
                            damage *= self.crown_tower_damage_multiplier
                    entity.take_damage(damage)

    def _apply_tornado_pull(self, entity: 'Entity', distance: float, dt: float, battle_state: 'BattleState') -> None:
        """Pull entity towards tornado center"""
        if distance == 0:
            return
        
        # Calculate pull vector towards tornado center
        dx = self.position.x - entity.position.x
        dy = self.position.y - entity.position.y
        
        # Normalize and apply pull force
        pull_distance = self.pull_force * dt
        
        # Don't pull past the center
        if pull_distance > distance:
            pull_distance = distance * 0.9  # Stop just short of center
        
        pull_x = (dx / distance) * pull_distance
        pull_y = (dy / distance) * pull_distance
        
        # Apply pull movement (air units can be pulled anywhere, ground units need walkable space)
        new_position = Position(entity.position.x + pull_x, entity.position.y + pull_y)
        
        if getattr(entity, 'is_air_unit', False) or battle_state.is_ground_position_walkable(new_position, entity):
            entity.position.x += pull_x
            entity.position.y += pull_y
    
    def _hitbox_overlaps_with_radius(self, entity: 'Entity') -> bool:
        """Check if entity's hitbox overlaps with area effect radius"""
        # Get entity collision radius (default to 0.5 tiles if not specified or None)
        if entity.card_stats and hasattr(entity.card_stats, 'collision_radius') and entity.card_stats.collision_radius is not None:
            entity_radius = entity.card_stats.collision_radius
        else:
            entity_radius = 0.5
        
        # Calculate distance between area center and entity center
        distance = entity.position.distance_to(self.position)
        
        # Check if area radius overlaps with entity hitbox
        return distance <= (self.radius + entity_radius)


@dataclass
class SpawnProjectile(Projectile):
    """Projectile that spawns units when it reaches target"""
    spawn_count: int = 3
    spawn_character: str = "Goblin"
    spawn_character_data: dict = None
    activation_delay: float = 0.0
    time_alive: float = 0.0
    
    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update projectile - move towards target and spawn units on impact"""
        if not self.is_alive:
            return

        self.time_alive += dt
        if self.time_alive < self.activation_delay:
            return
        
        # Move towards target
        distance = self.position.distance_to(self.target_position)
        if distance <= self.travel_speed * dt:
            # Reached target - spawn units and deal splash damage
            self._spawn_units(battle_state)
            self._deal_splash_damage(battle_state)
            self.is_alive = False
        else:
            self._move_towards(self.target_position, dt)
    
    def _spawn_units(self, battle_state: 'BattleState') -> None:
        """Spawn units at target position"""
        import math
        import random
        
        if not self.spawn_character_data:
            return
        
        spawn_stats = troop_from_character_data(
            self.spawn_character,
            self.spawn_character_data,
            elixir=0,
            rarity=self.spawn_character_data.get("rarity", "Common"),
        )
        
        # Spawn units in a small radius around target
        spawn_radius = 1.0  # tiles
        
        for _ in range(self.spawn_count):
            # Random position around the target location
            angle = random.random() * 2 * math.pi
            distance = random.random() * spawn_radius
            spawn_x = self.target_position.x + distance * math.cos(angle)
            spawn_y = self.target_position.y + distance * math.sin(angle)
            
            # Create and spawn the unit
            battle_state._spawn_troop(Position(spawn_x, spawn_y), self.player_id, spawn_stats)


@dataclass
class RollingProjectile(Entity):
    """Rolling projectiles that spawn at location and roll forward (Log, Barbarian Barrel)"""
    travel_speed: float = 200.0
    projectile_range: float = 10.0  # tiles
    spawn_delay: float = 0.65  # seconds
    spawn_character: str = None
    spawn_character_data: dict = None
    radius_y: float = 0.6  # Height of rolling hitbox
    knockback_distance: float = 1.5
    # Optional custom direction (unit vector); used by Bowler boulder
    target_direction_x: Optional[float] = None
    target_direction_y: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Use range field from Entity as rolling radius
        self.rolling_radius = self.range
    
    # State tracking
    time_alive: float = 0.0
    distance_traveled: float = 0.0
    hit_entities: set = field(default_factory=set)  # Track entities hit (can only hit once)
    has_spawned_character: bool = False
    
    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update rolling projectile - wait for spawn delay, then roll forward"""
        if not self.is_alive:
            return
        
        self.time_alive += dt
        
        # Wait for spawn delay before starting to roll
        if self.time_alive < self.spawn_delay:
            return
        
        # Roll forward at constant speed
        roll_distance = self.travel_speed / 60.0 * dt  # Convert tiles/min to tiles/sec
        self.distance_traveled += roll_distance
        
        # Determine roll direction
        if self.target_direction_x is not None and self.target_direction_y is not None:
            # Use custom direction (for Bowler)
            self.position.x += self.target_direction_x * roll_distance
            self.position.y += self.target_direction_y * roll_distance
        else:
            # Default direction (towards enemy side for Log/Barbarian Barrel)
            if self.player_id == 0:  # Blue player rolls towards red side (positive Y)
                self.position.y += roll_distance
            else:  # Red player rolls towards blue side (negative Y)
                self.position.y -= roll_distance
        
        # Check if reached max range
        if self.distance_traveled >= self.projectile_range:
            # Spawn character if applicable (Barbarian Barrel)
            if self.spawn_character and not self.has_spawned_character:
                self._spawn_character(battle_state)
            self.is_alive = False
            return
        
        # Deal damage to entities in rectangular hitbox
        self._deal_rolling_damage(battle_state)
    
    def _deal_rolling_damage(self, battle_state: 'BattleState') -> None:
        """Deal damage to ground units in rolling path (rectangular hitbox)"""
        # Create a copy of entities list to avoid RuntimeError when dictionary changes during iteration
        entities_copy = list(battle_state.entities.values())
        for entity in entities_copy:
            if (entity.player_id == self.player_id or 
                not entity.is_alive or 
                entity.id in self.hit_entities or
                entity == self):
                continue
            
            # Skip air units (Log only hits ground)
            if getattr(entity, 'is_air_unit', False):
                continue
            
            # Check if entity is in rectangular rolling hitbox with entity collision radius
            if self._hitbox_overlaps_with_rolling_path(entity):
                # Hit the entity
                entity.take_damage(self.damage)
                self.hit_entities.add(entity.id)
                
                # Apply knockback effect (Log pushes units backward)
                self._apply_knockback(entity, battle_state)

    def _apply_knockback(self, entity: 'Entity', battle_state: 'BattleState') -> None:
        """Apply knockback effect - pushes unit away from Log and resets attack"""
        # Buildings cannot be knocked back or stunned; they only take damage
        if isinstance(entity, Building):
            return

        # Reset attack cooldown (stunned briefly)
        if hasattr(entity, 'attack_cooldown'):
            entity.attack_cooldown = max(entity.attack_cooldown, 0.5)
        
        # Physical knockback - push unit away from Log's rolling direction
        knockback_distance = self.knockback_distance  # tiles
        
        # Determine knockback direction based on movement direction
        if self.target_direction_x is not None and self.target_direction_y is not None:
            # Push along projectile's travel vector (Bowler angled push)
            candidate_x = entity.position.x + self.target_direction_x * knockback_distance
            candidate_y = entity.position.y + self.target_direction_y * knockback_distance
        else:
            if self.player_id == 0:  # Blue player Log rolling toward red side
                # Push units further toward red side (positive Y)
                candidate_x = entity.position.x
                candidate_y = entity.position.y + knockback_distance
            else:  # Red player Log rolling toward blue side
                # Push units further toward blue side (negative Y)
                candidate_x = entity.position.x
                candidate_y = entity.position.y - knockback_distance
        
        # Slight random horizontal displacement for realism
        import random
        horizontal_variance = random.uniform(-0.3, 0.3)
        candidate_x += horizontal_variance

        # Keep within arena bounds
        candidate_x = max(0.5, min(17.5, candidate_x))
        candidate_y = max(0.5, min(31.5, candidate_y))
        candidate = Position(candidate_x, candidate_y)
        if battle_state.is_ground_position_walkable(candidate, entity):
            entity.position = candidate
    
    def _hitbox_overlaps_with_rolling_path(self, entity: 'Entity') -> bool:
        """Check if entity's hitbox overlaps with rolling projectile path"""
        # Get entity collision radius (default to 0.5 tiles if not specified or None)
        if entity.card_stats and hasattr(entity.card_stats, 'collision_radius') and entity.card_stats.collision_radius is not None:
            entity_radius = entity.card_stats.collision_radius
        else:
            entity_radius = 0.5
        
        # Calculate distance components
        dx = abs(entity.position.x - self.position.x)
        dy = abs(entity.position.y - self.position.y)
        
        # Check if entity hitbox overlaps with rectangular rolling area
        return dx <= (self.rolling_radius + entity_radius) and dy <= (self.radius_y + entity_radius)
    
    def _spawn_character(self, battle_state: 'BattleState') -> None:
        """Spawn character at end of roll (Barbarian Barrel)"""
        if not self.spawn_character_data:
            return
        
        import math

        spawn_stats = troop_from_character_data(
            self.spawn_character,
            self.spawn_character_data,
            elixir=0,
            rarity=self.spawn_character_data.get("rarity", "Common"),
        )
        
        # Spawn character at current position
        battle_state._spawn_troop(Position(self.position.x, self.position.y), self.player_id, spawn_stats)
        self.has_spawned_character = True


@dataclass
class TimedExplosive(Entity):
    """Entity that explodes after a countdown timer (death bombs, balloon bombs)"""
    explosion_timer: float = 3.0
    explosion_radius: float = 1.5
    explosion_damage: float = 600.0
    death_spawn_name: Optional[str] = None
    death_spawn_count: int = 0
    death_spawn_data: Optional[dict] = None
    time_alive: float = 0.0
    
    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update timed explosive - countdown and explode"""
        if not self.is_alive:
            return
            
        self.time_alive += dt
        
        # Check if timer expired
        if self.time_alive >= self.explosion_timer:
            self._explode(battle_state)
            self.is_alive = False
    
    def _explode(self, battle_state: 'BattleState') -> None:
        """Deal explosion damage to entities in radius using hitbox collision"""
        for entity in list(battle_state.entities.values()):
            if entity.player_id == self.player_id or not entity.is_alive or entity == self:
                continue
            
            # Use hitbox-based collision detection for explosion
            if self._hitbox_overlaps_with_explosion(entity):
                entity.take_damage(self.explosion_damage)

        # Optional chained death spawn (e.g. Skeleton Barrel container -> Skeletons).
        if self.death_spawn_name and self.death_spawn_count > 0:
            self._spawn_death_units(battle_state)

    def _spawn_death_units(self, battle_state: 'BattleState') -> None:
        """Spawn units around this explosive when configured."""
        import math
        import random

        from .factory.dynamic_factory import troop_from_character_data, troop_from_values

        spawn_stats = battle_state.card_loader.get_card(self.death_spawn_name)
        if not spawn_stats and self.death_spawn_data:
            spawn_stats = troop_from_character_data(
                self.death_spawn_name,
                self.death_spawn_data,
                elixir=0,
                rarity=self.death_spawn_data.get("rarity", "Common"),
            )
        if not spawn_stats:
            spawn_stats = troop_from_values(
                self.death_spawn_name,
                hitpoints=100,
                damage=25,
                speed_tiles_per_min=60.0,
                range_tiles=1.0,
                sight_range_tiles=5.0,
                hit_speed_ms=1000,
                collision_radius_tiles=0.5,
            )

        for _ in range(self.death_spawn_count):
            angle = random.random() * 2 * math.pi
            distance = random.random() * 0.7
            spawn_x = self.position.x + distance * math.cos(angle)
            spawn_y = self.position.y + distance * math.sin(angle)
            battle_state._spawn_troop(Position(spawn_x, spawn_y), self.player_id, spawn_stats)
    
    def _hitbox_overlaps_with_explosion(self, entity: 'Entity') -> bool:
        """Check if entity's hitbox overlaps with explosion radius"""
        # Get entity collision radius (default to 0.5 tiles if not specified or None)
        if entity.card_stats and hasattr(entity.card_stats, 'collision_radius') and entity.card_stats.collision_radius is not None:
            entity_radius = entity.card_stats.collision_radius
        else:
            entity_radius = 0.5
        
        # Calculate distance between explosion center and entity center
        distance = entity.position.distance_to(self.position)
        
        # Check if explosion radius overlaps with entity hitbox
        return distance <= (self.explosion_radius + entity_radius)


@dataclass
class Graveyard(Entity):
    """Entity that periodically spawns skeletons in an area"""
    spawn_interval: float = 0.5
    max_skeletons: int = 20
    spawn_radius: float = 2.5
    duration: float = 10.0
    skeleton_data: dict = None
    time_alive: float = 0.0
    time_since_spawn: float = 0.0
    skeletons_spawned: int = 0
    
    def update(self, dt: float, battle_state: 'BattleState') -> None:
        """Update graveyard - spawn skeletons periodically"""
        if not self.is_alive:
            return
            
        self.time_alive += dt
        self.time_since_spawn += dt
        
        # Check if duration expired
        if self.time_alive >= self.duration:
            self.is_alive = False
            return
        
        # Spawn skeleton if it's time and haven't reached max
        if (self.time_since_spawn >= self.spawn_interval and 
            self.skeletons_spawned < self.max_skeletons):
            self._spawn_skeleton(battle_state)
            self.time_since_spawn = 0.0
            self.skeletons_spawned += 1
    
    def _spawn_skeleton(self, battle_state: 'BattleState') -> None:
        """Spawn a skeleton at random position in radius"""
        import math
        import random
        
        if not self.skeleton_data:
            return
        
        # Create skeleton stats
        skeleton_stats = troop_from_character_data(
            "Skeleton",
            self.skeleton_data,
            elixir=0,
            rarity=self.skeleton_data.get("rarity", "Common"),
        )
        
        # Random position within spawn radius
        angle = random.random() * 2 * math.pi
        distance = random.random() * self.spawn_radius
        spawn_x = self.position.x + distance * math.cos(angle)
        spawn_y = self.position.y + distance * math.sin(angle)
        
        # Spawn the skeleton
        battle_state._spawn_troop(Position(spawn_x, spawn_y), self.player_id, skeleton_stats)
