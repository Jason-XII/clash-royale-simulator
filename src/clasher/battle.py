from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
import copy
import json

from .entities import Entity, Troop, Building
from .arena import TileGrid, Position
from .spells import SPELL_REGISTRY
from .mechanics.shared.death_effects import DeathSpawn

# What `BattleState` should do:
# - at first, initialize princess towers and king towers for both sides
# - handle entity spawning
# - update entities at every tick, remove dead entities
# - end the game when time is appropriate


@dataclass
class BattleState:
    def __init__(self):
        self.entities = {}
        self.players = []
        self.arena = TileGrid()

        self.time = 0.0
        self.tick = 0
        self.dt = 1 / 30  # 33ms per tick (~30 FPS)

        self.game_over = False
        self.winner = None

        self.next_entity_id = 1

    def _spawn_entity(self, entity):
        """Spawn any type of entity"""
        entity.battle_state = self
        self.entities[self.next_entity_id] = entity
        self.next_entity_id += 1
        entity.on_spawn()
    
    def __post_init__(self) -> None:
        """Initialize battle state"""
        self._create_towers()
        self._starting_total_tower_hp = {
            0: self.players[0].king_tower_hp + self.players[0].left_tower_hp + self.players[0].right_tower_hp,
            1: self.players[1].king_tower_hp + self.players[1].left_tower_hp + self.players[1].right_tower_hp,
        }
    
    def _create_towers(self) -> None:
        """Create tower entities for both players"""
        princess_data = self._load_princess_tower_character_data()
        princess_hitpoints = princess_data.get("hitpoints", 1400) if princess_data else 1400
        princess_range_tiles = (princess_data.get("range", 7500) / 1000.0) if princess_data else 7.5
        princess_sight_tiles = (princess_data.get("sightRange", 7500) / 1000.0) if princess_data else 7.5
        princess_hit_speed_ms = princess_data.get("hitSpeed", 800) if princess_data else 800
        princess_deploy_ms = princess_data.get("deployTime", 800) if princess_data else 800
        princess_collision_tiles = (princess_data.get("collisionRadius", 1000) / 1000.0) if princess_data else 1.0
        princess_projectile = princess_data.get("projectileData", {}) if princess_data else {}
        princess_damage = princess_projectile.get("damage", 50)
        princess_projectile_speed = princess_projectile.get("speed", 600)
        princess_target_type = princess_data.get("tidTarget", "TID_TARGETS_AIR_AND_GROUND") if princess_data else "TID_TARGETS_AIR_AND_GROUND"

        tower_stats = building_from_values(
            name="Tower",
            hitpoints=princess_hitpoints,
            damage=princess_damage,
            range_tiles=princess_range_tiles,
            sight_range_tiles=princess_sight_tiles,
            hit_speed_ms=princess_hit_speed_ms,
            deploy_time_ms=princess_deploy_ms,
            collision_radius_tiles=princess_collision_tiles,
            lifetime_ms=None,
            elixir=0,
            rarity="Common",
            projectile_speed=princess_projectile_speed,
            projectile_damage=princess_damage,
            target_type=princess_target_type,
            raw_overrides={"level": 11},
        )

        king_stats = building_from_values(
            name="KingTower",
            hitpoints=4824,
            damage=109,
            range_tiles=7.0,
            sight_range_tiles=7.0,
            hit_speed_ms=1000,
            deploy_time_ms=1000,
            collision_radius_tiles=1.0,
            lifetime_ms=None,
            elixir=0,
            rarity="Common",
            projectile_speed=600,
            projectile_damage=109,
            target_type="TID_TARGETS_AIR_AND_GROUND",
            raw_overrides={"level": 1},
        )
        
        # Player 0 towers (blue) - create new Position objects to avoid sharing references
        blue_left = Position(self.arena.BLUE_LEFT_TOWER.x, self.arena.BLUE_LEFT_TOWER.y)
        blue_right = Position(self.arena.BLUE_RIGHT_TOWER.x, self.arena.BLUE_RIGHT_TOWER.y)
        blue_king = Position(self.arena.BLUE_KING_TOWER.x, self.arena.BLUE_KING_TOWER.y)
        self._spawn_entity(Building, blue_left, 0, tower_stats)
        self._spawn_entity(Building, blue_right, 0, tower_stats)
        self._spawn_entity(Building, blue_king, 0, king_stats)
        
        # Player 1 towers (red) - create new Position objects to avoid sharing references
        red_left = Position(self.arena.RED_LEFT_TOWER.x, self.arena.RED_LEFT_TOWER.y)
        red_right = Position(self.arena.RED_RIGHT_TOWER.x, self.arena.RED_RIGHT_TOWER.y)
        red_king = Position(self.arena.RED_KING_TOWER.x, self.arena.RED_KING_TOWER.y)
        self._spawn_entity(Building, red_left, 1, tower_stats)
        self._spawn_entity(Building, red_right, 1, tower_stats)
        self._spawn_entity(Building, red_king, 1, king_stats)

    def _load_princess_tower_character_data(self) -> Optional[dict]:
        """Load Princess Tower baseline stats from support-card data in gamedata."""
        try:
            with open(self.card_loader.data_file, "r") as f:
                spells = json.load(f).get("items", {}).get("spells", [])
            for entry in spells:
                if entry.get("name") == "King_PrincessTowers":
                    data = entry.get("statCharacterData")
                    if isinstance(data, dict):
                        return data
        except Exception:
            return None
        return None
    
    def step(self, speed_factor: float = 1.0) -> None:
        """Advance battle by one tick"""
        if self.game_over:
            return
        
        dt = self.dt * speed_factor
        self.time += dt
        self.tick += 1
        
        # Update elixir modes
        if self.time >= 180.0 and not self.double_elixir:
            self.double_elixir = True
        if self.time >= self.overtime_start_time and not self.overtime:
            self.overtime = True
        if self.time >= 240.0 and not self.triple_elixir:
            self.triple_elixir = True
        
        # Regenerate elixir
        base_regen = 2.8
        if self.triple_elixir:
            base_regen = 0.93
        elif self.double_elixir:
            base_regen = 1.4
        
        for player in self.players:
            player.regenerate_elixir(dt, base_regen)
        
        # Update all entities
        for entity in list(self.entities.values()):
            entity.update(dt, self)

        # Resolve simple body collision to reduce unit stacking.
        self._resolve_troop_collisions()
        
        # Remove dead entities
        self._cleanup_dead_entities()
        
        # Check win conditions
        self._check_win_conditions()
    
    def deploy_card(self, player_id: int, card_name: str, position: Position) -> bool:
        """Deploy a card at the given position"""
        player = self.players[player_id]
        resolved_name = resolve_card_name(card_name, self.card_loader.load_card_definitions())
        card_stats = self.card_loader.get_card(resolved_name)

        if not card_stats or not player.can_play_card(card_name, card_stats):
            print("Cannot play card: ", player.can_play_card(card_name, card_stats))
            return False

        is_spell = resolved_name in SPELL_REGISTRY
        spell_obj = SPELL_REGISTRY.get(resolved_name) if is_spell else None
        # Miner can be deployed almost anywhere (except blocked/tower tiles).
        if resolved_name == "Miner":
            tile_pos = (int(position.x), int(position.y))
            if not self.arena.is_valid_position(position):
                return False
            if tile_pos in self.arena.BLOCKED_TILES:
                return False
            if self.arena.is_tower_tile(position, self):
                return False
        else:
            if not self.arena.can_deploy_at(position, player_id, self, is_spell, spell_obj):
                return False

        # Special deployment validation for Royal Recruits (center 6 tiles only)
        if resolved_name == 'RoyalRecruits' and not (6 <= position.x <= 11):
            return False

        if not is_spell:
            # Buildings are solid obstacles; do not allow overlapping deployment.
            probe_radius = getattr(card_stats, "collision_radius", 0.5) or 0.5
            if self.is_position_occupied_by_building(position, probe_radius):
                return False

        # Play the card
        if not player.play_card(card_name, card_stats):
            return False

        # Check if it's a spell
        if resolved_name in SPELL_REGISTRY:
            spell = SPELL_REGISTRY[resolved_name]
            spell.cast(self, player_id, position)
        else:
            # Spawn troop or building based on card type (robust to missing/None fields)
            card_type = getattr(card_stats, "card_type", None)
            card_type_str = str(card_type).lower() if card_type is not None else ""
            if card_type_str == "building":
                self._spawn_entity(Building, position, player_id, card_stats)
            else:
                self._spawn_troop(position, player_id, card_stats)
        
        return True
    
    def _spawn_troop(self, position: Position, player_id: int, card_stats: CardStatsCompat) -> None:
        """Spawn a troop entity (handles both single troops and swarms)"""
        # Guard: if this card is actually a building, route to building spawner
        ctype = getattr(card_stats, "card_type", None)
        ctype_str = str(ctype).lower() if ctype is not None else ""
        if ctype_str == "building":
            self._spawn_entity(Building, position, player_id, card_stats)
            return
        
        # Check if this is a swarm card (spawns multiple units)
        summon_count = getattr(card_stats, 'summon_count', None) or 1
        summon_radius = getattr(card_stats, 'summon_radius', None) or 0.5  # Default 0.5 tiles
        
        # Check for mixed swarms (like Goblin Gang)
        second_count = getattr(card_stats, 'summon_character_second_count', None) or 0
        second_data = getattr(card_stats, 'summon_character_second_data', None)
        
        if summon_count > 1 or second_count > 0:
            # Spawn swarm units in a circle around the target position
            # summon_radius is already converted to tiles in data loading
            self._spawn_swarm_troops(position, player_id, card_stats, summon_count, summon_radius, second_count, second_data)
        else:
            # Spawn single unit at exact position
            self._spawn_single_troop(position, player_id, card_stats)
    
    def _spawn_single_troop(self, position: Position, player_id: int, card_stats: CardStatsCompat) -> None:
        """Spawn a single troop entity"""
        # Get speed value (tiles/min from card stats)
        speed = card_stats.speed or 60.0  # Default to 60 tiles/min if not specified
        
        # Determine if this is an air unit (based on target_type or known list)
        air_units = ['Minions', 'MinionHorde', 'Balloon', 'SkeletonBalloon', 'BabyDragon', 
                    'InfernoDragon', 'ElectroDragon', 'SkeletonDragons', 'MegaMinion', 'DartBarrell',
                     'Bats', 'LavaHound']
        is_air_unit = (getattr(card_stats, 'name', '') in air_units) or (
            getattr(card_stats, 'target_type', '') == 'TID_TARGETS_AIR'
        )
        
        # Use level-scaled stats for hitpoints and damage
        scaled_hp = card_stats.scaled_hitpoints or 100
        scaled_damage = card_stats.scaled_damage or 10
        
        troop = Troop(
            id=self.next_entity_id,
            position=Position(position.x, position.y),  # Create new position object
            player_id=player_id,
            card_stats=card_stats,
            hitpoints=scaled_hp,
            max_hitpoints=scaled_hp,
            damage=scaled_damage,
            range=card_stats.range or 100,
            sight_range=card_stats.sight_range or 500,
            speed=speed,
            is_air_unit=is_air_unit
        )

        troop.deploy_delay_remaining = max(0.0, (getattr(card_stats, "deploy_time", 0) or 0) / 1000.0)
        troop.attack_cooldown = max(troop.attack_cooldown, (getattr(card_stats, "load_time", 0) or 0) / 1000.0)
        
        # Set battle reference
        troop.battle_state = self
        
        # Attach mechanics from factory definition if available
        try:
            defn_name = getattr(card_stats, 'name', None)
            card_def = self.card_loader.get_card_definition(defn_name) if defn_name else None
            if card_def and getattr(card_def, 'mechanics', None):
                troop.mechanics = [copy.deepcopy(m) for m in card_def.mechanics]
                for mech in troop.mechanics:
                    mech.on_attach(troop)
                if troop.mechanics:
                    print(f"[Attach] {defn_name}: {len(troop.mechanics)} mechanic(s)")
        except Exception as e:
            print(f"[Warn] Failed attaching mechanics for {getattr(card_stats, 'name', 'Unknown')}: {e}")
        
        self.entities[self.next_entity_id] = troop
        self.next_entity_id += 1
        
        # Fire spawn hooks
        troop.on_spawn()
    
    def _spawn_swarm_troops(self, center_pos: Position, player_id: int, card_stats: CardStatsCompat, count: int, radius: float, second_count: int = 0, second_data: dict = None) -> None:
        """Spawn multiple troop entities in a circle around center position"""
        import math
        import random
        
        total_units = count + second_count
        
        # Check for special deployment patterns
        is_royal_recruits = card_stats.name in ['RoyalRecruits', 'RoyalRecruits_Chess']
        
        # Check if this is a mixed swarm with front/back positioning (like Goblin Gang)
        has_front_back = (count > 0 and second_count > 0 and 
                         getattr(card_stats, 'summon_character_data', None) and
                         getattr(card_stats, 'summon_character_second_data', None))
        
        if is_royal_recruits:
            # Royal Recruits spawn in a horizontal line across battlefield
            self._spawn_royal_recruits_line(center_pos, player_id, card_stats, count)

        elif has_front_back:
            # Spawn in front/back formation for mixed swarms
            self._spawn_front_back_formation(center_pos, player_id, card_stats, count, second_count, second_data, radius)
        else:
            # Regular circular formation
            # Spawn primary units 
            for i in range(count):
                self._spawn_unit_at_angle(center_pos, player_id, card_stats, i, total_units, radius)
            
            # Spawn secondary units if available
            if second_count > 0 and second_data:
                unit_name = second_data.get("name", card_stats.name + "_Secondary")
                second_card_stats = self._create_card_stats_from_data(second_data, unit_name)
                
                for i in range(second_count):
                    self._spawn_unit_at_angle(center_pos, player_id, second_card_stats, count + i, total_units, radius)
    
    def _spawn_unit_at_angle(self, center_pos: Position, player_id: int, card_stats: CardStatsCompat, index: int, total_count: int, radius: float) -> None:
        """Spawn a single unit at a specific angle in the swarm formation"""
        import math
        import random
        
        # Calculate position in circle
        if total_count == 1:
            # Single unit at center
            spawn_x = center_pos.x
            spawn_y = center_pos.y
        else:
            angle = (2 * math.pi * index) / total_count
            # Add some randomness to avoid perfect circles
            angle_variance = random.uniform(-0.3, 0.3)
            radius_variance = random.uniform(0.7, 1.3)
            
            spawn_x = center_pos.x + (radius * radius_variance) * math.cos(angle + angle_variance)
            spawn_y = center_pos.y + (radius * radius_variance) * math.sin(angle + angle_variance)
        
        # Validate and snap position to playable area
        spawn_position = Position(spawn_x, spawn_y)
        spawn_position = self._snap_to_valid_position(spawn_position, player_id)
        
        # Get unit properties
        speed = card_stats.speed or 60.0
        
        # Determine if this is an air unit
        air_units = ['Minions', 'MinionHorde', 'Balloon', 'SkeletonBalloon', 'BabyDragon',
                     'InfernoDragon', 'ElectroDragon', 'SkeletonDragons', 'MegaMinion', 'DartBarrell',
                     'Bats', 'LavaHound']
        is_air_unit = card_stats.name in air_units
        
        # Use level-scaled stats for hitpoints and damage
        scaled_hp = card_stats.scaled_hitpoints or card_stats.hitpoints or 100
        scaled_damage = card_stats.scaled_damage or card_stats.damage or 10
        
        troop = Troop(
            id=self.next_entity_id,
            position=spawn_position,
            player_id=player_id,
            card_stats=card_stats,
            hitpoints=scaled_hp,
            max_hitpoints=scaled_hp,
            damage=scaled_damage,
            range=card_stats.range or 100,
            sight_range=card_stats.sight_range or 500,
            speed=speed,
            is_air_unit=is_air_unit
        )

        troop.deploy_delay_remaining = max(0.0, (getattr(card_stats, "deploy_time", 0) or 0) / 1000.0)
        troop.attack_cooldown = max(troop.attack_cooldown, (getattr(card_stats, "load_time", 0) or 0) / 1000.0)
        
        # Attach mechanics if using compat wrapper with card_definition
        if hasattr(card_stats, 'card_definition') and card_stats.card_definition:
            troop.mechanics = [copy.deepcopy(m) for m in card_stats.card_definition.mechanics]
        # Set battle reference
        troop.battle_state = self
        
        # Attach mechanics from factory definition if available
        try:
            defn_name = getattr(card_stats, 'name', None)
            card_def = self.card_loader.get_card_definition(defn_name) if defn_name else None
            if card_def and getattr(card_def, 'mechanics', None):
                troop.mechanics = [copy.deepcopy(m) for m in card_def.mechanics]
                for mech in troop.mechanics:
                    mech.on_attach(troop)
                if troop.mechanics:
                    print(f"[Attach] {defn_name}: {len(troop.mechanics)} mechanic(s)")
        except Exception as e:
            print(f"[Warn] Failed attaching mechanics for {getattr(card_stats, 'name', 'Unknown')}: {e}")
        
        self.entities[self.next_entity_id] = troop
        self.next_entity_id += 1
        
        # Fire spawn hooks
        troop.on_spawn()
    
    def _spawn_front_back_formation(self, center_pos: Position, player_id: int, card_stats: CardStatsCompat, front_count: int, back_count: int, back_data: dict, radius: float) -> None:
        """Spawn units in front/back formation (melee in front, ranged in back)"""
        import math
        import random
        
        # Create card stats for both unit types
        # Primary units (front) - use actual name from summonCharacterData
        primary_data = getattr(card_stats, 'summon_character_data', {})
        primary_name = primary_data.get("name", card_stats.name)
        front_card_stats = self._create_card_stats_from_data(primary_data, primary_name)
        
        # Secondary units (back) - use actual name from summonCharacterSecondData  
        back_name = back_data.get("name", card_stats.name + "_Secondary")
        back_card_stats = self._create_card_stats_from_data(back_data, back_name)
        
        # Determine formation based on player direction
        # Blue player (0): front = towards enemy (positive Y), back = towards own side (negative Y)
        # Red player (1): front = towards enemy (negative Y), back = towards own side (positive Y)
        
        front_offset = 0.6  # tiles in front of center
        back_offset = 0.8   # tiles behind center
        
        if player_id == 0:  # Blue player
            front_y = center_pos.y + front_offset  # Towards red side
            back_y = center_pos.y - back_offset    # Towards blue side
        else:  # Red player  
            front_y = center_pos.y - front_offset  # Towards blue side
            back_y = center_pos.y + back_offset    # Towards red side
        
        # Spawn front units (melee) in a line
        for i in range(front_count):
            # Spread horizontally across the front
            if front_count == 1:
                front_x = center_pos.x
            else:
                spread = radius * 1.2  # Slightly wider spread for front line
                front_x = center_pos.x + (i - (front_count - 1) / 2) * (spread * 2 / (front_count - 1))
            
            # Add slight randomness
            front_x += random.uniform(-0.2, 0.2)
            front_y_final = front_y + random.uniform(-0.1, 0.1)
            
            front_pos = Position(front_x, front_y_final)
            front_pos = self._snap_to_valid_position(front_pos, player_id)
            
            self._spawn_unit_at_position(front_pos, player_id, front_card_stats)
        
        # Spawn back units (ranged) in a line
        for i in range(back_count):
            # Spread horizontally across the back
            if back_count == 1:
                back_x = center_pos.x
            else:
                spread = radius * 1.2
                back_x = center_pos.x + (i - (back_count - 1) / 2) * (spread * 2 / (back_count - 1))
            
            # Add slight randomness
            back_x += random.uniform(-0.2, 0.2)
            back_y_final = back_y + random.uniform(-0.1, 0.1)
            
            back_pos = Position(back_x, back_y_final)
            back_pos = self._snap_to_valid_position(back_pos, player_id)
            
            self._spawn_unit_at_position(back_pos, player_id, back_card_stats)
    
    # def _spawn_royal_recruits_line(self, center_pos: Position, player_id: int, card_stats: CardStatsCompat, count: int) -> None:
    #     """Spawn Royal Recruits in a horizontal line across the battlefield, avoiding towers"""
    #     # Royal Recruits: 6 units spaced 2.5 tiles apart, center at deploy position
    #     spacing = 2.5  # tiles between each recruit
    #
    #     # Get tower-blocked X ranges for this Y coordinate
    #     blocked_ranges = self.arena.get_tower_blocked_x_ranges(center_pos.y, self)
    #
    #     # Calculate initial line positions
    #     total_width = (count - 1) * spacing
    #     leftmost_x = center_pos.x - (total_width / 2)
    #
    #     # Generate all recruit X positions
    #     recruit_positions = []
    #     for i in range(count):
    #         recruit_x = leftmost_x + (i * spacing)
    #         recruit_positions.append(recruit_x)
    #
    #     # Check if any positions would overlap with towers
    #     needs_adjustment = False
    #     for recruit_x in recruit_positions:
    #         for x_min, x_max in blocked_ranges:
    #             if x_min <= recruit_x <= x_max:
    #                 needs_adjustment = True
    #                 break
    #         if needs_adjustment:
    #             break
    #
    #     # If line overlaps with towers, find alternative positioning
    #     if needs_adjustment:
    #         recruit_positions = self._find_safe_recruit_positions(center_pos, count, spacing, blocked_ranges)
    #
    #     # Ensure all positions are within arena bounds
    #     recruit_positions = [max(0.5, min(17.5, x)) for x in recruit_positions]
    #
    #     # Spawn each recruit
    #     for recruit_x in recruit_positions:
    #         recruit_pos = Position(recruit_x, center_pos.y)
    #         recruit_pos = self._snap_to_valid_position(recruit_pos, player_id)
    #         self._spawn_unit_at_position(recruit_pos, player_id, card_stats)
    def _spawn_royal_recruits_line(self, center_pos: Position, player_id: int, card_stats: CardStatsCompat,
                                   count: int) -> None:
        """Spawn Royal Recruits in fixed lane positions with 3-3, 4-2, or 2-4 split"""
        # Fixed x positions for left and right lanes (evenly spaced within each lane)
        left_positions = [2.0, 4.0, 6.0, 8.0]  # up to 4 on left
        right_positions = [10.0, 12.0, 14.0, 16.0]  # up to 4 on right

        # Determine split based on tap x
        if center_pos.x < 7.5:
            left_xs = left_positions[:4]
            right_xs = right_positions[:2]
        elif center_pos.x > 10.5:
            left_xs = left_positions[:2]
            right_xs = right_positions[:4]
        else:
            left_xs = left_positions[:3]
            right_xs = right_positions[:3]
        blocked_ranges = self.arena.get_tower_blocked_x_ranges(center_pos.y, self)
        all_xs = []
        for x in left_xs + right_xs:
            blocked = any(x_min <= x <= x_max for x_min, x_max in blocked_ranges)
            if blocked:
                # Nudge away from blocked range
                for x_min, x_max in blocked_ranges:
                    if x_min <= x <= x_max:
                        x = x_max + 1.0 if x > (x_min + x_max) / 2 else x_min - 1.0
                        break
            all_xs.append(max(0.5, min(17.5, x)))

        for x in all_xs:
            pos = Position(x, center_pos.y)
            self._spawn_single_troop(pos, player_id, card_stats)
    
    def _find_safe_recruit_positions(self, center_pos: Position, count: int, spacing: float, blocked_ranges: List[Tuple[float, float]]) -> List[float]:
        """Find safe X positions for Royal Recruits that avoid tower collisions"""
        
        # Create list of all blocked X coordinates
        blocked_x_coords = set()
        for x_min, x_max in blocked_ranges:
            # Add all positions in blocked range with 0.5 precision
            x = x_min
            while x <= x_max:
                blocked_x_coords.add(round(x * 2) / 2)  # Round to nearest 0.5
                x += 0.5
        
        # Find all safe X positions across the arena
        safe_positions = []
        for x_half in range(1, 36):  # 0.5 to 17.5 in 0.5 increments
            x = x_half / 2.0
            if x not in blocked_x_coords and 0.5 <= x <= 17.5:
                safe_positions.append(x)
        
        # If we have enough safe positions, try to maintain spacing
        if len(safe_positions) >= count:
            # Try to find positions with good spacing starting from center
            selected_positions = []
            
            # Find safe position closest to center
            center_candidates = [pos for pos in safe_positions if abs(pos - center_pos.x) <= 1.0]
            if not center_candidates:
                center_candidates = safe_positions
            
            center_safe = min(center_candidates, key=lambda x: abs(x - center_pos.x))
            selected_positions.append(center_safe)
            
            # For remaining positions, try to maintain spacing while staying safe
            while len(selected_positions) < count:
                best_candidate = None
                best_score = float('inf')
                
                for candidate in safe_positions:
                    if candidate in selected_positions:
                        continue
                    
                    # Score based on distance from ideal spacing positions
                    min_spacing_score = float('inf')
                    for existing_pos in selected_positions:
                        spacing_distance = abs(abs(candidate - existing_pos) - spacing)
                        min_spacing_score = min(min_spacing_score, spacing_distance)
                    
                    # Prefer positions that maintain good spacing
                    if min_spacing_score < best_score:
                        best_score = min_spacing_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    selected_positions.append(best_candidate)
                else:
                    # If no good candidate, just pick the first available
                    for pos in safe_positions:
                        if pos not in selected_positions:
                            selected_positions.append(pos)
                            break
            
            positions = selected_positions
        else:
            # Not enough safe positions, use what we have
            positions = safe_positions[:count]
        
        # Ensure we have exactly the right count
        while len(positions) < count:
            # Add fallback positions at arena edges
            for x in [0.5, 1.0, 17.0, 17.5, 16.5, 16.0]:
                if x not in positions and x not in blocked_x_coords:
                    positions.append(x)
                    if len(positions) >= count:
                        break
        
        # Sort and return exact count
        positions.sort()
        return positions[:count]
    
    def _spawn_unit_at_position(self, position: Position, player_id: int, card_stats: CardStatsCompat) -> None:
        """Spawn a single unit at a specific position"""
        # Get unit properties
        speed = card_stats.speed or 60.0
        
        # Determine if this is an air unit
        air_units = ['Minions', 'MinionHorde', 'Balloon', 'SkeletonBalloon', 'BabyDragon',
                     'InfernoDragon', 'ElectroDragon', 'SkeletonDragons', 'MegaMinion', 'DartBarrell',
                     'Bats', 'LavaHound']
        is_air_unit = card_stats.name in air_units
        
        # Use level-scaled stats for hitpoints and damage
        scaled_hp = card_stats.scaled_hitpoints or card_stats.hitpoints or 100
        scaled_damage = card_stats.scaled_damage or card_stats.damage or 10
        
        troop = Troop(
            id=self.next_entity_id,
            position=position,
            player_id=player_id,
            card_stats=card_stats,
            hitpoints=scaled_hp,
            max_hitpoints=scaled_hp,
            damage=scaled_damage,
            range=card_stats.range or 100,
            sight_range=card_stats.sight_range or 500,
            speed=speed,
            is_air_unit=is_air_unit
        )

        troop.deploy_delay_remaining = max(0.0, (getattr(card_stats, "deploy_time", 0) or 0) / 1000.0)
        troop.attack_cooldown = max(troop.attack_cooldown, (getattr(card_stats, "load_time", 0) or 0) / 1000.0)
        troop.battle_state = self

        try:
            defn_name = getattr(card_stats, 'name', None)
            card_def = self.card_loader.get_card_definition(defn_name) if defn_name else None
            if card_def and getattr(card_def, 'mechanics', None):
                troop.mechanics = [copy.deepcopy(m) for m in card_def.mechanics]
                for mech in troop.mechanics:
                    mech.on_attach(troop)
        except Exception as e:
            print(f"[Warn] Failed attaching mechanics for {getattr(card_stats, 'name', 'Unknown')}: {e}")

        self.entities[self.next_entity_id] = troop
        self.next_entity_id += 1
        troop.on_spawn()
    
    def _snap_to_valid_position(self, position: Position, player_id: int) -> Position:
        """Snap position to nearest valid playable area"""
        # Check if position is already valid (walkable and not on a tower)
        if self.arena.is_walkable(position) and not self.arena.is_tower_tile(position, self):
            return position
        
        # Try to find nearest valid position within reasonable distance
        search_radius = 2.0  # tiles
        best_position = position
        min_distance = float('inf')
        
        # Search in a grid around the original position
        for x_offset in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
            for y_offset in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                test_x = position.x + x_offset
                test_y = position.y + y_offset
                test_pos = Position(test_x, test_y)
                
                # Check bounds first
                if not self.arena.is_valid_position(test_pos):
                    continue
                
                # Check if walkable and not on tower
                if self.arena.is_walkable(test_pos) and not self.arena.is_tower_tile(test_pos, self):
                    distance = position.distance_to(test_pos)
                    if distance < min_distance:
                        min_distance = distance
                        best_position = test_pos
        
        # If no valid position found nearby, clamp to arena bounds and find closest walkable
        if min_distance == float('inf'):
            # Clamp to arena bounds
            clamped_x = max(0.5, min(17.5, position.x))
            clamped_y = max(0.5, min(31.5, position.y))
            clamped_pos = Position(clamped_x, clamped_y)
            
            # If clamped position is walkable and not on tower, use it
            if self.arena.is_walkable(clamped_pos) and not self.arena.is_tower_tile(clamped_pos, self):
                best_position = clamped_pos
            else:
                # Fallback: move towards arena center until we find walkable area
                center_x, center_y = 9.0, 16.0
                dx = center_x - clamped_x
                dy = center_y - clamped_y
                
                for step in [0.5, 1.0, 1.5, 2.0]:
                    fallback_x = clamped_x + dx * step * 0.1
                    fallback_y = clamped_y + dy * step * 0.1
                    fallback_pos = Position(fallback_x, fallback_y)
                    
                    if (self.arena.is_valid_position(fallback_pos) and 
                        self.arena.is_walkable(fallback_pos) and
                        not self.arena.is_tower_tile(fallback_pos, self)):
                        best_position = fallback_pos
                        break
        
        return best_position
    
    def _create_card_stats_from_data(self, unit_data: dict, name: str) -> CardStatsCompat:
        """Create CardStatsCompat from raw unit data (for secondary units in mixed swarms)."""
        if unit_data:
            rarity = unit_data.get("rarity", "Common")
            return troop_from_character_data(name, unit_data, elixir=0, rarity=rarity)

        # Fallback minimal definition if no data is provided
        return troop_from_values(
            name,
            hitpoints=100,
            damage=10,
            speed_tiles_per_min=60.0,
            range_tiles=1.0,
            sight_range_tiles=5.0,
            hit_speed_ms=1000,
            collision_radius_tiles=0.5,
        )
    

    
    def _cleanup_dead_entities(self) -> None:
        """Remove dead entities from the game and handle death spawns"""
        dead_ids = [eid for eid, entity in self.entities.items() if not entity.is_alive]
        
        # Handle death spawns before removing entities (skip if handled by mechanics)
        for eid in dead_ids:
            entity = self.entities[eid]
            if isinstance(entity, Troop) and getattr(entity.card_stats, 'death_spawn_character', None):
                has_mechanic_spawn = any(isinstance(m, DeathSpawn) for m in getattr(entity, 'mechanics', []))
                if not has_mechanic_spawn:
                    self._spawn_death_units(entity)
        
        # Update player state for dead towers before removing entities
        for eid in dead_ids:
            entity = self.entities[eid]
            if isinstance(entity, Building):
                player = self.players[entity.player_id]
                pos = entity.position
                
                # Set tower HP to 0 when entity dies
                if entity.player_id == 0:  # Blue player
                    if (pos.x == self.arena.BLUE_KING_TOWER.x and 
                        pos.y == self.arena.BLUE_KING_TOWER.y):
                        player.king_tower_hp = 0
                    elif (pos.x == self.arena.BLUE_LEFT_TOWER.x and 
                          pos.y == self.arena.BLUE_LEFT_TOWER.y):
                        player.left_tower_hp = 0
                        self._activate_king_tower(0)
                    elif (pos.x == self.arena.BLUE_RIGHT_TOWER.x and 
                          pos.y == self.arena.BLUE_RIGHT_TOWER.y):
                        player.right_tower_hp = 0
                        self._activate_king_tower(0)
                else:  # Red player
                    if (pos.x == self.arena.RED_KING_TOWER.x and 
                        pos.y == self.arena.RED_KING_TOWER.y):
                        player.king_tower_hp = 0
                    elif (pos.x == self.arena.RED_LEFT_TOWER.x and 
                          pos.y == self.arena.RED_LEFT_TOWER.y):
                        player.left_tower_hp = 0
                        self._activate_king_tower(1)
                    elif (pos.x == self.arena.RED_RIGHT_TOWER.x and 
                          pos.y == self.arena.RED_RIGHT_TOWER.y):
                        player.right_tower_hp = 0
                        self._activate_king_tower(1)
        
        # Remove dead entities
        for eid in dead_ids:
            del self.entities[eid]
    
    def _spawn_death_units(self, troop: Troop) -> None:
        """Spawn death units when a troop dies"""
        death_spawn_name = troop.card_stats.death_spawn_character
        death_spawn_count = troop.card_stats.death_spawn_count or 1
        
        # Get death spawn card stats using the factory-driven loader
        death_spawn_stats = self.card_loader.get_card(death_spawn_name)

        if not death_spawn_stats and getattr(troop.card_stats, 'death_spawn_character_data', None):
            death_spawn_stats = troop_from_character_data(
                death_spawn_name,
                troop.card_stats.death_spawn_character_data,
                elixir=0,
                rarity="Common",
            )

        if not death_spawn_stats:
            death_spawn_stats = troop_from_values(
                death_spawn_name,
                hitpoints=100,
                damage=25,
                speed_tiles_per_min=60.0,
                range_tiles=1.0,
                sight_range_tiles=5.0,
                hit_speed_ms=1000,
                collision_radius_tiles=0.5,
            )
        
        # Spawn multiple units in a small radius around the death position
        spawn_radius = 0.5  # tiles
        
        for _ in range(death_spawn_count):
            # Random position around the death location
            angle = random.random() * 2 * 3.14159
            distance = random.random() * spawn_radius
            spawn_x = troop.position.x + distance * math.cos(angle)
            spawn_y = troop.position.y + distance * math.sin(angle)
            
            # Create and spawn the death unit
            self._spawn_troop(Position(spawn_x, spawn_y), troop.player_id, death_spawn_stats)
    
    def _check_win_conditions(self) -> None:
        """Check if game should end"""
        # Update player tower HP from entities
        self._update_tower_hp()
        
        # Check if king towers are destroyed
        for i, player in enumerate(self.players):
            if not player.is_alive():
                self.game_over = True
                self.winner = 1 - i
                return
        
        player0_crowns = self.players[0].get_crown_count()
        player1_crowns = self.players[1].get_crown_count()

        # End of 5:00 match timer. If tied, enter sudden death.
        if self.time >= self.sudden_death_start_time and not self.sudden_death:
            if player0_crowns != player1_crowns:
                self.game_over = True
                self.winner = 0 if player0_crowns > player1_crowns else 1
                return
            self.sudden_death = True
            self._sudden_death_crowns = (player0_crowns, player1_crowns)

        # Sudden death: first crown advantage wins instantly.
        if self.sudden_death:
            if player0_crowns != player1_crowns:
                self.game_over = True
                self.winner = 0 if player0_crowns > player1_crowns else 1
                return

            # Long-running tie fallback: tiebreaker by total tower HP damage dealt.
            if self.time >= self.tiebreaker_time:
                p0_damage = self._tower_damage_dealt_by_player(0)
                p1_damage = self._tower_damage_dealt_by_player(1)
                if p0_damage > p1_damage:
                    self.winner = 0
                elif p1_damage > p0_damage:
                    self.winner = 1
                else:
                    self.winner = None
                self.game_over = True
    
    def _update_tower_hp(self) -> None:
        """Update player tower HP from building entities"""
        for entity in self.entities.values():
            if isinstance(entity, Building):
                player = self.players[entity.player_id]
                pos = entity.position
                
                # Update HP based on tower position (compare coordinates)
                if entity.player_id == 0:  # Blue player
                    if (pos.x == self.arena.BLUE_KING_TOWER.x and 
                        pos.y == self.arena.BLUE_KING_TOWER.y):
                        player.king_tower_hp = entity.hitpoints
                    elif (pos.x == self.arena.BLUE_LEFT_TOWER.x and 
                          pos.y == self.arena.BLUE_LEFT_TOWER.y):
                        player.left_tower_hp = entity.hitpoints
                    elif (pos.x == self.arena.BLUE_RIGHT_TOWER.x and 
                          pos.y == self.arena.BLUE_RIGHT_TOWER.y):
                        player.right_tower_hp = entity.hitpoints
                else:  # Red player
                    if (pos.x == self.arena.RED_KING_TOWER.x and 
                        pos.y == self.arena.RED_KING_TOWER.y):
                        player.king_tower_hp = entity.hitpoints
                    elif (pos.x == self.arena.RED_LEFT_TOWER.x and 
                          pos.y == self.arena.RED_LEFT_TOWER.y):
                        player.left_tower_hp = entity.hitpoints
                    elif (pos.x == self.arena.RED_RIGHT_TOWER.x and 
                          pos.y == self.arena.RED_RIGHT_TOWER.y):
                        player.right_tower_hp = entity.hitpoints
    
    def get_state_summary(self) -> Dict:
        """Get current battle state summary"""
        return {
            "time": self.time,
            "tick": self.tick,
            "entities": len(self.entities),
            "players": [
                {
                    "elixir": p.elixir,
                    "crowns": p.get_crown_count(),
                    "king_hp": p.king_tower_hp,
                    "left_hp": p.left_tower_hp,
                    "right_hp": p.right_tower_hp,
                    "next_card": p.get_next_card(),
                }
                for p in self.players
            ],
            "game_over": self.game_over,
            "winner": self.winner
        }

    def _get_king_tower_entity(self, player_id: int) -> Optional[Building]:
        king_pos = self.arena.BLUE_KING_TOWER if player_id == 0 else self.arena.RED_KING_TOWER
        for entity in self.entities.values():
            if isinstance(entity, Building) and entity.player_id == player_id:
                if entity.position.x == king_pos.x and entity.position.y == king_pos.y:
                    return entity
        return None

    def _activate_king_tower(self, player_id: int) -> None:
        king = self._get_king_tower_entity(player_id)
        if king is not None:
            king._tower_active = True

    def _tower_damage_dealt_by_player(self, player_id: int) -> float:
        enemy_id = 1 - player_id
        current_enemy_hp = (
            self.players[enemy_id].king_tower_hp
            + self.players[enemy_id].left_tower_hp
            + self.players[enemy_id].right_tower_hp
        )
        start_enemy_hp = self._starting_total_tower_hp.get(enemy_id, current_enemy_hp)
        return max(0.0, start_enemy_hp - current_enemy_hp)

    def is_position_occupied_by_building(
        self,
        position: Position,
        mover_radius: float = 0.5,
        ignore_building_id: Optional[int] = None,
    ) -> bool:
        """Return True when a position overlaps any live building footprint."""
        for entity in self.entities.values():
            if not isinstance(entity, Building) or not entity.is_alive:
                continue
            if ignore_building_id is not None and entity.id == ignore_building_id:
                continue
            building_radius = getattr(entity.card_stats, "collision_radius", 1.0) or 1.0
            if position.distance_to(entity.position) < (building_radius + mover_radius) * 0.95:
                return True
        return False

    def is_ground_position_walkable(
        self,
        position: Position,
        mover: Optional[Entity] = None,
    ) -> bool:
        """Ground movement validator including arena terrain and building footprints."""
        if not self.arena.is_walkable(position):
            return False
        mover_radius = 0.5
        if mover is not None:
            mover_radius = getattr(mover.card_stats, "collision_radius", 0.5) or 0.5
        return not self.is_position_occupied_by_building(position, mover_radius)

    def _resolve_troop_collisions(self) -> None:
        """Simple separation pass to reduce troop stacking."""
        troops = [e for e in self.entities.values() if isinstance(e, Troop) and e.is_alive]
        for i in range(len(troops)):
            a = troops[i]
            if getattr(a, "is_air_unit", False):
                continue
            ra = max(0.2, getattr(a.card_stats, "collision_radius", 0.5) or 0.5)
            for j in range(i + 1, len(troops)):
                b = troops[j]
                if getattr(b, "is_air_unit", False):
                    continue
                rb = max(0.2, getattr(b.card_stats, "collision_radius", 0.5) or 0.5)
                dx = b.position.x - a.position.x
                dy = b.position.y - a.position.y
                dist = math.hypot(dx, dy)
                same_team = a.player_id == b.player_id
                min_dist = (ra + rb) * (0.8 if same_team else 0.65)
                if dist == 0:
                    dist = 0.001
                    dx, dy = 0.001, 0.0
                if dist < min_dist:
                    overlap = (min_dist - dist) / 2.0
                    ux, uy = dx / dist, dy / dist
                    new_a = Position(a.position.x - ux * overlap, a.position.y - uy * overlap)
                    new_b = Position(b.position.x + ux * overlap, b.position.y + uy * overlap)
                    if self.is_ground_position_walkable(new_a, a):
                        a.position = new_a
                    if self.is_ground_position_walkable(new_b, b):
                        b.position = new_b
