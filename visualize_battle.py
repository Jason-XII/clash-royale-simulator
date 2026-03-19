#!/usr/bin/env python3
"""
Real-time Battle Visualization

Shows the battlefield with:
- Arena layout with river and towers
- Unit positions and movement
- Health bars and stats
- Real-time battle progression
"""

import pygame
import sys
import time
import json
import os
import math
from typing import Dict, List, Tuple
from src.clasher.engine import BattleEngine
from src.clasher.arena import Position

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 100, 100)
BLUE = (100, 100, 255)
GREEN = (100, 255, 100)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
YELLOW = (255, 255, 100)
PURPLE = (255, 100, 255)
CYAN = (100, 255, 255)
ORANGE = (255, 165, 0)

# Screen settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900
# Make tiles square: 18 wide × 32 tall
TILE_SIZE = 22  # Square tiles
ARENA_WIDTH = 18 * TILE_SIZE   # 396 pixels
ARENA_HEIGHT = 32 * TILE_SIZE  # 704 pixels
ARENA_X = 50
ARENA_Y = 50

class BattleVisualizer:
    def __init__(self):
        self.paused = False
        self.speed = 1
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Clash Royale Battle Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 32)
        
        # Load hitbox data (these are radii in tile units)
        try:
            with open('hitboxes.json', 'r') as f:
                self.hitboxes = json.load(f)
        except FileNotFoundError:
            print("Warning: hitboxes.json not found, using default values")
            self.hitboxes = {
                "Knight": 0.5, "Archers": 0.5, "King's Tower": 1.4, "Princess Tower": 1.0
            }
        
        # Battle setup
        self.engine = BattleEngine()
        self.battle = self.engine.create_battle()
        
        # Visualization settings - square tiles
        self.tile_size = TILE_SIZE  # Square tiles
        self.tile_width = TILE_SIZE
        self.tile_height = TILE_SIZE
        
        # Auto-deploy for testing
        self.scheduled_deploys = []
        self.setup_test_battle()

        
    def setup_test_battle(self):
        """Set up a test battle with units"""
        print("Setting up test battle with corrected arena dimensions...")
        
        # Give players more elixir for testing
        self.battle.players[0].hand = ['Princess']
        self.battle.players[0].deck = ['Princess']
        self.battle.players[1].hand = ['GoblinGang']
        self.battle.players[1].deck = ['GoblinGang']


        self.battle.players[0].elixir = 10.0
        self.battle.deploy_card(0, 'Princess', Position(6, 4))

        # Deploy Goblin Gang after 3 seconds
        # self.scheduled_deploys = [(6.0, 1, 'GoblinGang', Position(4, 28))]

        
        # Deploy some units for visualization using corrected positions
        deployments = [
        ]

        for player_id, card, pos, desc in deployments:
            result = self.battle.deploy_card(player_id, card, pos)
            print(f"  {desc}: {result}")
            # Reset elixir for testing
            self.battle.players[player_id].elixir = 10.0
        
        # Enable coordinate display for debugging
        self.show_coords = True
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int(ARENA_X + x * self.tile_size)
        screen_y = int(ARENA_Y + y * self.tile_size)
        return screen_x, screen_y
    
    def draw_arena(self):
        """Draw the arena background with correct Clash Royale dimensions"""
        # Arena background
        arena_rect = pygame.Rect(ARENA_X, ARENA_Y, ARENA_WIDTH, ARENA_HEIGHT)
        pygame.draw.rect(self.screen, GREEN, arena_rect)
        pygame.draw.rect(self.screen, BLACK, arena_rect, 2)
        
        # Add deployment zone tints
        self.draw_deployment_zones()
        
        # River is 2 tiles tall, centered at y=15-16 (middle of 32-tile arena)
        river_y1 = ARENA_Y + 15 * self.tile_size
        river_y2 = ARENA_Y + 17 * self.tile_size
        river_rect = pygame.Rect(ARENA_X, river_y1, ARENA_WIDTH, river_y2 - river_y1)
        pygame.draw.rect(self.screen, CYAN, river_rect)
        pygame.draw.rect(self.screen, (0, 150, 200), river_rect, 2)
        
        # Bridges are 3 tiles wide
        bridge_width = 3 * self.tile_size
        bridge_height = river_y2 - river_y1
        
        # Left bridge at x=2,3,4 (3 tiles wide, centered at x=3)
        left_bridge_x = ARENA_X + 2 * self.tile_size
        left_bridge_rect = pygame.Rect(left_bridge_x, river_y1, bridge_width, bridge_height)
        pygame.draw.rect(self.screen, DARK_GRAY, left_bridge_rect)
        pygame.draw.rect(self.screen, BLACK, left_bridge_rect, 2)
        
        # Right bridge at x=13,14,15 (3 tiles wide, centered at x=14)
        right_bridge_x = ARENA_X + 13 * self.tile_size
        right_bridge_rect = pygame.Rect(right_bridge_x, river_y1, bridge_width, bridge_height)
        pygame.draw.rect(self.screen, DARK_GRAY, right_bridge_rect)
        pygame.draw.rect(self.screen, BLACK, right_bridge_rect, 2)
        
        # Mark unplayable areas (outer tiles on top/bottom rows)
        # Top row: only middle 6 tiles (x=6-11) are playable
        for x in range(18):
            if x < 6 or x > 11:  # Unplayable tiles
                unplay_rect = pygame.Rect(ARENA_X + x * self.tile_size, ARENA_Y, 
                                        self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, (180, 180, 180), unplay_rect)
                pygame.draw.rect(self.screen, DARK_GRAY, unplay_rect, 1)
        
        # Bottom row: only middle 6 tiles (x=6-11) are playable  
        for x in range(18):
            if x < 6 or x > 11:  # Unplayable tiles
                unplay_rect = pygame.Rect(ARENA_X + x * self.tile_size, 
                                        ARENA_Y + 31 * self.tile_size,
                                        self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, (180, 180, 180), unplay_rect)
                pygame.draw.rect(self.screen, DARK_GRAY, unplay_rect, 1)
        
        # Draw square tile grid
        for x in range(19):  # 0-18 vertical lines for 18 columns
            line_x = ARENA_X + x * self.tile_size
            pygame.draw.line(self.screen, (0, 150, 0), 
                           (line_x, ARENA_Y), (line_x, ARENA_Y + ARENA_HEIGHT), 1)
        
        for y in range(33):  # 0-32 horizontal lines for 32 rows
            line_y = ARENA_Y + y * self.tile_size
            pygame.draw.line(self.screen, (0, 150, 0),
                           (ARENA_X, line_y), (ARENA_X + ARENA_WIDTH, line_y), 1)
        
        # Add coordinate labels for clarity
        if hasattr(self, 'show_coords') and self.show_coords:
            for y in range(0, 32, 4):
                coord_text = self.small_font.render(str(y), True, BLACK)
                self.screen.blit(coord_text, (ARENA_X - 25, ARENA_Y + y * self.tile_size))
            
            for x in range(0, 18, 3):
                coord_text = self.small_font.render(str(x), True, BLACK)
                self.screen.blit(coord_text, (ARENA_X + x * self.tile_size, ARENA_Y - 20))
    
    def draw_deployment_zones(self):
        """Draw colored tints for deployment zones"""
        if not hasattr(self, 'battle') or not self.battle:
            return
            
        # Get deployment zones for both players
        blue_zones = self.battle.arena.get_deploy_zones(0, self.battle)
        red_zones = self.battle.arena.get_deploy_zones(1, self.battle)
        
        # Create a grid to track which tiles belong to which player
        tile_ownership = {}  # (x, y) -> set of player_ids
        
        # Mark blue zones (excluding blocked tiles)
        for x1, y1, x2, y2 in blue_zones:
            for x in range(int(x1), int(x2)):
                for y in range(int(y1), int(y2)):
                    # Skip blocked tiles entirely - don't add them to tile_ownership
                    if self.battle.arena.is_blocked_tile(x, y):
                        continue
                    if (x, y) not in tile_ownership:
                        tile_ownership[(x, y)] = set()
                    tile_ownership[(x, y)].add(0)  # Blue player
        
        # Mark red zones (excluding blocked tiles)
        for x1, y1, x2, y2 in red_zones:
            for x in range(int(x1), int(x2)):
                for y in range(int(y1), int(y2)):
                    # Skip blocked tiles entirely - don't add them to tile_ownership
                    if self.battle.arena.is_blocked_tile(x, y):
                        continue
                    if (x, y) not in tile_ownership:
                        tile_ownership[(x, y)] = set()
                    tile_ownership[(x, y)].add(1)  # Red player
        
        # Draw tints based on ownership
        for (tile_x, tile_y), owners in tile_ownership.items():
            # Skip blocked tiles - they will be drawn gray instead
            if self.battle.arena.is_blocked_tile(tile_x, tile_y):
                continue
                
            # Skip if both players can deploy (no tint)
            if len(owners) > 1:
                continue
                
            # Determine tint color - same for all playable areas
            if 0 in owners:  # Blue only
                tint_color = (100, 150, 255, 120)  # More visible blue tint
            elif 1 in owners:  # Red only
                tint_color = (255, 150, 100, 120)  # More visible red tint
            else:
                continue
            
            # Draw the tint
            screen_x = ARENA_X + tile_x * TILE_SIZE
            screen_y = ARENA_Y + tile_y * TILE_SIZE
            
            tint_rect = pygame.Rect(screen_x, screen_y, TILE_SIZE, TILE_SIZE)
            s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            s.fill(tint_color)
            self.screen.blit(s, tint_rect)
        
        # Draw blocked tiles in gray (no red/blue tints)
        for tile_x in range(18):  # Arena width
            for tile_y in range(32):  # Arena height
                if self.battle.arena.is_blocked_tile(tile_x, tile_y):
                    screen_x = ARENA_X + tile_x * TILE_SIZE
                    screen_y = ARENA_Y + tile_y * TILE_SIZE
                    
                    # Draw gray blocked tile
                    blocked_rect = pygame.Rect(screen_x, screen_y, TILE_SIZE, TILE_SIZE)
                    s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    s.fill((128, 128, 128, 160))  # Semi-transparent gray
                    self.screen.blit(s, blocked_rect)
                    
                    # Add darker border to make it more visible
                    pygame.draw.rect(self.screen, (64, 64, 64), blocked_rect, 1)
    
    def draw_towers(self):
        """Draw tower positions"""
        tower_positions = [
            # Player 0 towers (blue) - bottom
            (9.0, 2.5, "King", BLUE),     # King tower centered at x=9 (middle of 18-wide arena)
            (3.5, 6.5, "Left", BLUE), 
            (14.5, 6.5, "Right", BLUE),   # Right tower at x=14.5 for better symmetry
            # Player 1 towers (red) - top
            (9.0, 29.5, "King", RED),     # King tower centered at x=9
            (3.5, 25.5, "Left", RED),
            (14.5, 25.5, "Right", RED)    # Right tower at x=14.5 for better symmetry
        ]
        
        for x, y, tower_type, color in tower_positions:
            screen_x, screen_y = self.world_to_screen(x, y)
            
            # Draw tower with circular hitbox visualization using actual hitbox data
            # Get radius from hitboxes.json (these are radius values in tile units)
            if tower_type == "King":
                hitbox_radius_tiles = self.hitboxes.get("King's Tower", 1.4)  # Default 1.4 tile radius
                visual_radius_tiles = hitbox_radius_tiles * 0.6  # Visual is smaller than hitbox
            else:
                hitbox_radius_tiles = self.hitboxes.get("Princess Tower", 1.0)  # Default 1.0 tile radius
                visual_radius_tiles = hitbox_radius_tiles * 0.6  # Visual is smaller than hitbox
            
            # Convert tile radius to screen pixels
            tower_radius = int(hitbox_radius_tiles * self.tile_size)
            visual_radius = int(visual_radius_tiles * self.tile_size)
            
            # Draw hitbox circle (light color)
            hitbox_color = (color[0]//2, color[1]//2, color[2]//2, 100)  # Semi-transparent
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), tower_radius, 2)
            
            # Draw tower visual (filled circle)
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), visual_radius)
            pygame.draw.circle(self.screen, BLACK, (screen_x, screen_y), visual_radius, 2)
            
            # Find tower entity for health
            for entity in self.battle.entities.values():
                if (hasattr(entity, 'card_stats') and entity.card_stats and 
                    abs(entity.position.x - x) < 1 and abs(entity.position.y - y) < 1):
                    
                    # Health bar above tower
                    health_ratio = entity.hitpoints / entity.max_hitpoints
                    bar_width = tower_radius * 2
                    bar_height = 4
                    bar_x = screen_x - bar_width // 2
                    bar_y = screen_y - tower_radius - 12
                    
                    # Health bar background
                    pygame.draw.rect(self.screen, BLACK, 
                                   (bar_x-1, bar_y-1, bar_width+2, bar_height+2))
                    # Health bar fill
                    health_color = GREEN if health_ratio > 0.6 else YELLOW if health_ratio > 0.3 else RED
                    pygame.draw.rect(self.screen, health_color,
                                   (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
                    
                    # HP text below tower
                    hp_text = self.small_font.render(f"{int(entity.hitpoints)}", True, BLACK)
                    hp_rect = hp_text.get_rect(center=(screen_x, screen_y + tower_radius + 12))
                    self.screen.blit(hp_text, hp_rect)
                    
                    # Tower type label
                    type_text = self.small_font.render(tower_type[:4], True, BLACK)
                    type_rect = type_text.get_rect(center=(screen_x, screen_y))
                    self.screen.blit(type_text, type_rect)
                    break
    
    def draw_entities(self):
        """Draw all entities (troops, projectiles, etc.)"""
        for entity in self.battle.entities.values():
            if not entity.is_alive:
                continue
            
            # Skip towers (drawn separately)
            if (hasattr(entity, 'card_stats') and entity.card_stats and 
                entity.card_stats.name in ['Tower', 'KingTower']):
                continue
            
            screen_x, screen_y = self.world_to_screen(entity.position.x, entity.position.y)
            
            # Determine color and shape based on player and type
            color = BLUE if entity.player_id == 0 else RED
            
            # Get entity type and handle special cases
            entity_name = "Unknown"
            entity_type = type(entity).__name__
            if hasattr(entity, 'card_stats') and entity.card_stats:
                entity_name = entity.card_stats.name
            
            # Special handling for spell entities
            if entity_type == "AreaEffect":
                # Draw area effect spell (circle on ground with timer)
                area_radius = int(entity.radius * self.tile_size)
                # Draw effect area
                pygame.draw.circle(self.screen, (255, 255, 0, 100), (screen_x, screen_y), area_radius, 3)
                # Draw timer indicator
                time_ratio = 1.0 - (entity.time_alive / entity.duration) if entity.duration > 0 else 0
                timer_color = (255, int(255 * time_ratio), 0)
                pygame.draw.circle(self.screen, timer_color, (screen_x, screen_y), 5)
                # Draw spell name
                spell_name = getattr(entity, 'spell_name', 'AREA')
                text_surface = self.small_font.render(spell_name.upper(), True, BLACK)
                text_rect = text_surface.get_rect(center=(screen_x, screen_y - 15))
                self.screen.blit(text_surface, text_rect)
                continue
            elif entity_type in ["Projectile", "SpawnProjectile"]:
                # Draw projectile (moving missile)
                projectile_radius = 4
                
                # Draw AoE radius at target position (like AreaEffect spells)
                if hasattr(entity, 'target_position') and hasattr(entity, 'splash_radius') and entity.splash_radius > 0:
                    target_screen_x, target_screen_y = self.world_to_screen(entity.target_position.x, entity.target_position.y)
                    aoe_radius_pixels = int(entity.splash_radius * self.tile_size)
                    # Draw AoE circle at target location (same style as AreaEffect)
                    pygame.draw.circle(self.screen, (255, 255, 0, 100), (target_screen_x, target_screen_y), aoe_radius_pixels, 3)
                    # Draw smaller target marker in center
                    pygame.draw.circle(self.screen, (255, 200, 0), (target_screen_x, target_screen_y), 3)
                
                # Draw projectile trail
                if hasattr(entity, 'target_position'):
                    start_x, start_y = self.world_to_screen(entity.position.x, entity.position.y)
                    end_x, end_y = self.world_to_screen(entity.target_position.x, entity.target_position.y)
                    pygame.draw.line(self.screen, (255, 255, 255, 128), (start_x, start_y), (end_x, end_y), 1)
                
                # Draw projectile
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), projectile_radius)
                pygame.draw.circle(self.screen, BLACK, (screen_x, screen_y), projectile_radius, 1)
                
                # Add projectile source name
                source_name = getattr(entity, 'source_name', getattr(entity, 'spell_name', 'PROJECTILE'))
                display_text = f"{source_name.upper()}"
                if hasattr(entity, 'splash_radius') and entity.splash_radius > 0:
                    display_text += f" ({entity.splash_radius:.1f}R)"
                
                spell_text = self.small_font.render(display_text, True, BLACK)
                text_rect = spell_text.get_rect(center=(screen_x, screen_y - 15))
                self.screen.blit(spell_text, text_rect)
                continue
            elif entity_type == "RollingProjectile":
                # Draw rolling projectile (rectangular hitbox, rolls forward)
                # Draw rectangular rolling area
                width = int(entity.rolling_radius * 2 * self.tile_size)
                height = int(entity.radius_y * 2 * self.tile_size)
                rect = pygame.Rect(screen_x - width//2, screen_y - height//2, width, height)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 2)
                
                # Show spawn delay countdown or rolling state
                if entity.time_alive < entity.spawn_delay:
                    countdown = entity.spawn_delay - entity.time_alive
                    status_text = f"{countdown:.1f}s"
                else:
                    status_text = "ROLLING"
                
                # Add spell name and status
                spell_name = getattr(entity, 'spell_name', 'ROLLING')
                spell_text = self.small_font.render(f"{spell_name.upper()}", True, BLACK)
                status_surface = self.small_font.render(status_text, True, BLACK)
                
                spell_rect = spell_text.get_rect(center=(screen_x, screen_y - 20))
                status_rect = status_surface.get_rect(center=(screen_x, screen_y - 5))
                
                self.screen.blit(spell_text, spell_rect)
                self.screen.blit(status_surface, status_rect)
                continue
            
            # Regular entities: All units have circular hitboxes - use actual hitbox data from JSON
            # Get hitbox radius from hitboxes.json (radius in tile units)
            hitbox_radius_tiles = self.hitboxes.get(entity_name, 0.5)  # Default 0.5 tile radius
            visual_radius_tiles = hitbox_radius_tiles * 0.7  # Visual is smaller than hitbox
            
            # Convert tile radius to screen pixels
            hitbox_radius = int(hitbox_radius_tiles * self.tile_size)
            visual_radius = int(visual_radius_tiles * self.tile_size)
            
            # Draw hitbox circle
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), hitbox_radius, 2)
            
            # Draw visual based on entity type
            if entity_name == "Knight":
                # Draw visual (square inside circle for distinction)
                pygame.draw.rect(self.screen, color, 
                               (screen_x - visual_radius//2, screen_y - visual_radius//2, 
                                visual_radius, visual_radius))
                pygame.draw.rect(self.screen, BLACK,
                               (screen_x - visual_radius//2, screen_y - visual_radius//2, 
                                visual_radius, visual_radius), 2)
            elif entity_name in ["Archers", "Archer"]:
                # Draw visual (triangle inside circle)
                points = [
                    (screen_x, screen_y - visual_radius),
                    (screen_x - visual_radius, screen_y + visual_radius//2),
                    (screen_x + visual_radius, screen_y + visual_radius//2)
                ]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.polygon(self.screen, BLACK, points, 2)
            else:
                # Generic units: circular visual
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), visual_radius)
                pygame.draw.circle(self.screen, BLACK, (screen_x, screen_y), visual_radius, 2)
                # Draw visual circle
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), visual_radius)
                pygame.draw.circle(self.screen, BLACK, (screen_x, screen_y), visual_radius, 2)
            
            # Draw AoE radius for ground troops with area damage (only for 0.5 seconds after attack)
            if (entity_type in ["Troop", "Building"] and 
                hasattr(entity, 'card_stats') and entity.card_stats):
                area_damage_radius = getattr(entity.card_stats, 'area_damage_radius', None)
                projectile_speed = getattr(entity.card_stats, 'projectile_speed', None)
                
                # Only show AoE for true melee units (not ranged units like Princess)
                # Exclude units that have projectile_speed (they're ranged units)
                is_melee_with_aoe = (area_damage_radius and area_damage_radius > 0 and 
                                    (not projectile_speed or projectile_speed == 0))
                
                if is_melee_with_aoe and hasattr(entity, 'last_attack_time'):
                    # Only show AoE circle for 0.5 seconds after attack
                    time_since_attack = self.battle.time - entity.last_attack_time
                    if 0 <= time_since_attack <= 0.5:
                        # Convert from game units to tiles to pixels
                        aoe_radius_tiles = area_damage_radius / 1000.0
                        aoe_radius_pixels = int(aoe_radius_tiles * self.tile_size)
                        
                        # Fade the circle over time (bright at start, fade to transparent)
                        alpha = int(255 * (1.0 - time_since_attack / 0.5))
                        
                        # Draw AoE circle (red for ground troops)
                        pygame.draw.circle(self.screen, (255, 100, 100), (screen_x, screen_y), aoe_radius_pixels, 3)
                        # Draw smaller center marker
                        pygame.draw.circle(self.screen, (200, 50, 50), (screen_x, screen_y), 3)
            
            # Health bar ABOVE the name tag (always above the unit)
            if hasattr(entity, 'hitpoints') and hasattr(entity, 'max_hitpoints'):
                health_ratio = entity.hitpoints / entity.max_hitpoints
                # Place bar above the unit's hitbox; name tag remains below
                bar_y_offset = hitbox_radius + 25  # a bit more spacing above the unit
                
                bar_width = int(self.tile_size * 0.8)
                bar_height = 3
                bar_x = screen_x - bar_width // 2
                bar_y = screen_y - bar_y_offset
                
                # Health bar background
                pygame.draw.rect(self.screen, BLACK,
                                 (bar_x-1, bar_y-1, bar_width+2, bar_height+2))
                # Health bar fill
                health_color = GREEN if health_ratio > 0.6 else YELLOW if health_ratio > 0.3 else RED
                pygame.draw.rect(self.screen, health_color,
                                 (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
            
            # Draw sight range circle (semi-transparent)
            if hasattr(entity, 'sight_range') and entity.sight_range:
                sight_radius = int(entity.sight_range * self.tile_size)
                # Draw sight range as a dotted circle
                for angle in range(0, 360, 15):  # Draw dotted circle
                    x = screen_x + int(sight_radius * math.cos(math.radians(angle)))
                    y = screen_y + int(sight_radius * math.sin(math.radians(angle)))
                    pygame.draw.circle(self.screen, (color[0]//2, color[1]//2, color[2]//2), (x, y), 3)
            
            # Red arrow - shows what troop is currently fixated on
            if hasattr(entity, 'target_id') and entity.target_id:
                target = self.battle.entities.get(entity.target_id)
                if target and target.is_alive:
                    distance_to_target = entity.position.distance_to(target.position)
                    
                    # Set arrow color based on troop team
                    arrow_color = BLUE if entity.player_id == 0 else RED
                    
                    # Determine what to point the arrow at based on what troop is fixated on
                    if distance_to_target <= entity.sight_range:
                        # Can see target - arrow points directly to what it's fixated on (troop or tower)
                        arrow_target_x, arrow_target_y = self.world_to_screen(target.position.x, target.position.y)
                    else:
                        # Can't see target - check if pathfinding to bridge/waypoint or just walking forward
                        if hasattr(entity, '_get_pathfind_target'):
                            pathfind_target = entity._get_pathfind_target(target)
                            
                            # If pathfinding target is different from just walking forward, show it
                            forward_y = entity.position.y + (3.0 if entity.player_id == 0 else -3.0)
                            forward_pos = (entity.position.x, forward_y)
                            pathfind_pos = (pathfind_target.x, pathfind_target.y)
                            
                            # Check if pathfinding target is significantly different from forward direction
                            if abs(pathfind_pos[0] - forward_pos[0]) > 1.0 or abs(pathfind_pos[1] - forward_pos[1]) > 1.0:
                                # Arrow points to specific pathfinding target (bridge, waypoint, etc.)
                                arrow_target_x, arrow_target_y = self.world_to_screen(pathfind_target.x, pathfind_target.y)
                            else:
                                # Arrow points forward (generic search)
                                arrow_target_x, arrow_target_y = self.world_to_screen(forward_pos[0], forward_pos[1])
                        else:
                            # No pathfinding - just point forward
                            forward_y = entity.position.y + (3.0 if entity.player_id == 0 else -3.0)
                            arrow_target_x, arrow_target_y = self.world_to_screen(entity.position.x, forward_y)
                    
                    # Draw the arrow
                    pygame.draw.line(self.screen, arrow_color, 
                                   (screen_x, screen_y), (arrow_target_x, arrow_target_y), 3)
                    # Draw circle at target
                    pygame.draw.circle(self.screen, arrow_color, (arrow_target_x, arrow_target_y), 8, 2)
            
            # Entity label (name tag) - draw below the unit
            label = self.small_font.render(entity_name, True, BLACK)
            label_rect = label.get_rect(center=(screen_x, screen_y + 25))
            self.screen.blit(label, label_rect)
    
    def draw_ui(self):
        """Draw UI elements"""
        # Panel background
        panel_rect = pygame.Rect(ARENA_X + ARENA_WIDTH + 20, ARENA_Y, 400, ARENA_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        
        ui_x = panel_rect.x + 10
        ui_y = panel_rect.y + 10
        line_height = 25
        
        # Title
        title = self.large_font.render("Battle Status", True, BLACK)
        self.screen.blit(title, (ui_x, ui_y))
        ui_y += 40
        
        # Time and tick
        time_text = self.font.render(f"Time: {self.battle.time:.1f}s", True, BLACK)
        self.screen.blit(time_text, (ui_x, ui_y))
        ui_y += line_height
        
        tick_text = self.font.render(f"Tick: {self.battle.tick}", True, BLACK)
        self.screen.blit(tick_text, (ui_x, ui_y))
        ui_y += line_height * 2
        
        # Player stats
        for i, player in enumerate(self.battle.players):
            color = BLUE if i == 0 else RED
            player_text = self.font.render(f"Player {i}:", True, color)
            self.screen.blit(player_text, (ui_x, ui_y))
            ui_y += line_height
            
            elixir_text = self.font.render(f"  Elixir: {player.elixir:.1f}/10", True, BLACK)
            self.screen.blit(elixir_text, (ui_x, ui_y))
            ui_y += line_height
            
            crowns_text = self.font.render(f"  Crowns: {player.get_crown_count()}", True, BLACK)
            self.screen.blit(crowns_text, (ui_x, ui_y))
            ui_y += line_height
            
            towers_text = self.font.render(f"  King: {int(player.king_tower_hp)}", True, BLACK)
            self.screen.blit(towers_text, (ui_x, ui_y))
            ui_y += line_height
            
            towers_text2 = self.font.render(f"  Towers: {int(player.left_tower_hp)}/{int(player.right_tower_hp)}", True, BLACK)
            self.screen.blit(towers_text2, (ui_x, ui_y))
            ui_y += line_height * 2
        
        # Entity count
        alive_entities = sum(1 for e in self.battle.entities.values() if e.is_alive)
        entities_text = self.font.render(f"Entities: {alive_entities}", True, BLACK)
        self.screen.blit(entities_text, (ui_x, ui_y))
        ui_y += line_height
        
        # Troops breakdown
        troops = {}
        for entity in self.battle.entities.values():
            if (entity.is_alive and hasattr(entity, 'card_stats') and 
                entity.card_stats and entity.card_stats.name not in ['Tower', 'KingTower']):
                name = entity.card_stats.name
                player = entity.player_id
                key = f"P{player} {name}"
                troops[key] = troops.get(key, 0) + 1
        
        if troops:
            troops_title = self.font.render("Active Troops:", True, BLACK)
            self.screen.blit(troops_title, (ui_x, ui_y))
            ui_y += line_height
            
            for troop, count in troops.items():
                troop_text = self.small_font.render(f"  {troop}: {count}", True, BLACK)
                self.screen.blit(troop_text, (ui_x, ui_y))
                ui_y += 20
        
        # Game state
        ui_y += line_height
        if self.battle.game_over:
            winner_text = self.font.render(f"WINNER: Player {self.battle.winner}!", True, GREEN)
            self.screen.blit(winner_text, (ui_x, ui_y))
        elif self.battle.double_elixir:
            elixir_text = self.font.render("DOUBLE ELIXIR!", True, PURPLE)
            self.screen.blit(elixir_text, (ui_x, ui_y))
        
        # Controls
        ui_y = panel_rect.bottom - 120
        controls_title = self.font.render("Controls:", True, BLACK)
        self.screen.blit(controls_title, (ui_x, ui_y))
        ui_y += line_height
        
        controls = [
            "SPACE: Pause/Resume",
            "R: Reset Battle", 
            "1-5: Speed (1x to 5x)",
            "ESC: Exit"
        ]
        
        for control in controls:
            control_text = self.small_font.render(control, True, BLACK)
            self.screen.blit(control_text, (ui_x, ui_y))
            ui_y += 20
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not getattr(self, 'paused', False)
                elif event.key == pygame.K_r:
                    # Reset battle
                    self.engine = BattleEngine()
                    self.battle = self.engine.create_battle()
                    self.setup_test_battle()
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    # Set speed multiplier
                    self.speed = event.key - pygame.K_0
        return True
    
    def run(self):
        """Main visualization loop"""
        print("🎮 Starting Battle Visualization")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset Battle")
        print("  1-5: Speed multiplier (1x to 5x)")
        print("  I: Toggle investigation mode (auto screenshots)")
        print("  S: Take manual screenshot")
        print("  P: Take pathfinding debug screenshot")
        print("  ESC: Exit")
        
        self.paused = False
        self.speed = 1
        running = True
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update battle
            if not self.paused and not self.battle.game_over:
                for _ in range(self.speed):
                    self.battle.step(speed_factor=1.0)
                    for deploy in self.scheduled_deploys[:]:
                        t, pid, card, pos = deploy
                        if self.battle.time >= t:
                            self.battle.deploy_card(pid, card, pos)
                            self.battle.players[pid].elixir = 10.0
                            self.scheduled_deploys.remove(deploy)
                # Investigation mode - take screenshots at intervals
                if getattr(self, 'investigation_mode', False):
                    if not hasattr(self, 'investigation_counter'):
                        self.investigation_counter = 0
                    self.investigation_counter += self.speed
                    
                    # Take screenshot every 30 ticks
                    if self.investigation_counter >= 30:
                        self.investigation_counter = 0
                        # Draw everything first
                        self.screen.fill(WHITE)
                        self.draw_arena()
                        self.draw_towers()
                        self.draw_entities()
                        self.draw_ui()
                        pygame.display.flip()
                        
                        # Take investigation screenshot
                        tick = getattr(self.battle, 'tick', 0)
                        filename = f"{self.investigation_folder}/tick_{tick:04d}.png"
                        self.take_screenshot(filename)
            
            # Draw everything
            self.screen.fill(WHITE)
            self.draw_arena()
            self.draw_towers()
            self.draw_entities()
            self.draw_ui()
            
            # Show pause indicator
            if self.paused:
                pause_text = self.large_font.render("PAUSED", True, RED)
                pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH//2, 30))
                self.screen.blit(pause_text, pause_rect)
            
            # Show speed indicator
            if self.speed > 1:
                speed_text = self.font.render(f"Speed: {self.speed}x", True, PURPLE)
                speed_rect = speed_text.get_rect(topleft=(10, 10))
                self.screen.blit(speed_text, speed_rect)
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS display
        
        pygame.quit()


class CustomBattle(BattleVisualizer):
    def __init__(self):
        super().__init__()

    def deploy(self, card_name, coordinates, player=0, interval=0):
        current_player = self.battle.players[player]
        current_player.hand = [card_name]
        current_player.deck = [card_name]
        current_player.elixir = 10
        if not interval:
            self.battle.deploy_card(player, card_name, Position(*coordinates))
        else:
            self.scheduled_deploys.append((interval, player, card_name, coordinates))

    def setup_test_battle(self):
        pass


if __name__ == "__main__":
    battle = CustomBattle()
    battle.deploy('Berserker', (8, 8))
    battle.run()
