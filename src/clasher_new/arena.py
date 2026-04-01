from dataclasses import dataclass
from .core import Position


@dataclass
class TileGrid:
    width, height = 18, 32
    tile_size: float = 100.0
    BLUE_KING_TOWER = Position(9.0, 3.0)     # King tower centered at x=9 (middle of 18-wide arena)
    BLUE_LEFT_TOWER = Position(3.5, 6.5)     # Left princess tower
    BLUE_RIGHT_TOWER = Position(14.5, 6.5)   # Right princess tower (corrected for better symmetry)
    RED_KING_TOWER = Position(9.0, 29.0)     # King tower centered at x=9
    RED_LEFT_TOWER = Position(3.5, 25.5)     # Left princess tower  
    RED_RIGHT_TOWER = Position(14.5, 25.5)   # Right princess tower (corrected for better symmetry)
    LEFT_BRIDGE = Position(3.5, 16.0)   # Left bridge center of center tile (tiles 2,3,4 -> center at 3.5)
    RIGHT_BRIDGE = Position(14.5, 16.0) # Right bridge center of center tile (tiles 13,14,15 -> center at 14.5)
    RIVER_Y1 = 15.0
    RIVER_Y2 = 16.0
    BLOCKED_TILES = [
        # Edge tiles next to river
        (0, 15), (0, 16), (1, 15), (1, 16),
        *[(i, j) for i in range(5, 13) for j in range(15, 17)], # (5, 15) to (12, 16)
        (16, 15), (16, 16), (17, 15), (17, 16),
        
        # Top row (y=0): 6 gray fences (0-5), 6 green king area (6-11), 6 gray fences (12-17)
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),           # Left 6 gray fence tiles
        (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0),     # Right 6 gray fence tiles
        
        # Bottom row (y=31): 6 gray fences (0-5), 6 green king area (6-11), 6 gray fences (12-17)
        (0, 31), (1, 31), (2, 31), (3, 31), (4, 31), (5, 31),     # Left 6 gray fence tiles
        (12, 31), (13, 31), (14, 31), (15, 31), (16, 31), (17, 31), # Right 6 gray fence tiles
    ]

    towers = [
        (BLUE_LEFT_TOWER, 1.5, 0),  # Princess tower, 3x3
        (BLUE_RIGHT_TOWER, 1.5, 0),  # Princess tower, 3x3
        (BLUE_KING_TOWER, 2.0, 0),  # King tower, 4x4
        (RED_LEFT_TOWER, 1.5, 1),  # Princess tower, 3x3
        (RED_RIGHT_TOWER, 1.5, 1),  # Princess tower, 3x3
        (RED_KING_TOWER, 2.0, 1)  # King tower, 4x4
    ]
    def is_valid_position(self, pos):
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height
    
    def is_blocked_tile(self, x: int, y: int) -> bool:
        return (x, y) in self.BLOCKED_TILES
    
    def is_walkable(self, pos: Position) -> bool:
        if not self.is_valid_position(pos): return False
        if self.is_blocked_tile(int(pos.x), int(pos.y)): return False
        if self.RIVER_Y1 <= pos.y <= self.RIVER_Y2:
            on_left_bridge = 2.0 <= pos.x < 5.0
            on_right_bridge = 13.0 <= pos.x < 16.0
            return on_left_bridge or on_right_bridge
        return True

    def _is_tower_alive(self, tower_pos: Position, player_id: int, battle_state) -> bool:
        """Check if tower at given position is still alive"""
        for entity in battle_state.entities.values():
            if (hasattr(entity, 'position') and
                entity.position.x == tower_pos.x and
                entity.position.y == tower_pos.y and
                getattr(entity, 'player_id', -1) == player_id):
                return entity.is_alive
        return False  # Tower not found, assume dead

    def _is_rolling_projectile_spell(self, spell_obj):
        """Check if spell is a rolling projectile that requires territory validation"""
        from .spells import RollingProjectileSpell
        return isinstance(spell_obj, RollingProjectileSpell)

    def is_tower_tile(self, pos: Position, battle_state=None) -> bool:
        """Check if position overlaps with any living tower's occupied area"""
        for tower_pos, radius, player_id in self.towers:
            # Check if tower is still alive (if battle_state provided)
            if battle_state:
                tower_alive = self._is_tower_alive(tower_pos, player_id, battle_state)
                if not tower_alive:
                    continue
            dx = abs(pos.x - tower_pos.x)
            dy = abs(pos.y - tower_pos.y)
            if dx <= radius and dy <= radius:
                return True

        return False
    
    def get_deploy_zones(self, player_id: int, battle_state=None):
        """Get valid deployment zones for a player (x1, y1, x2, y2)
        Expands to include bridge areas and 4 tiles back when towers are destroyed"""
        zones = []
        
        if player_id == 0:  # Player 0 (bottom half)
            # Basic deployment zone (bottom half, excluding river)
            zones.append((0, 1, self.width, self.RIVER_Y1))  # y=1 to y=14
            zones.append((6, 0, 12, 6))  # Behind blue king: x=6-11, y=0-5 (6 tiles behind own king, including edge row)
            if battle_state:
                # If red left tower is destroyed, blue player can spawn on left half of arena and 4 tiles back
                if battle_state.players[1].left_tower_hp <= 0:
                    zones.append((0, self.RIVER_Y2 + 1, 9, self.RIVER_Y2 + 5))  # Left half: x=0-8, y=17-20
                # If red right tower is destroyed, blue player can spawn on right half of arena and 4 tiles back  
                if battle_state.players[1].right_tower_hp <= 0:
                    zones.append((9, self.RIVER_Y2 + 1, self.width, self.RIVER_Y2 + 5))  # Right half: x=9-17, y=17-20
        else:  # Player 1 (top half)
            zones.append((0, self.RIVER_Y2 + 1, self.width, 31))  # y=17 to y=30
            zones.append((6, 26, 12, 32))  # Behind red king: x=6-11, y=26-31 (6 tiles behind own king, including edge row)
            if battle_state:
                # If blue left tower is destroyed, red player can spawn on left half of arena and 4 tiles back
                if battle_state.players[0].left_tower_hp <= 0:
                    zones.append((0, self.RIVER_Y1 - 4, 9, self.RIVER_Y1))  # Left half: x=0-8, y=11-14
                # If blue right tower is destroyed, red player can spawn on right half of arena and 4 tiles back
                if battle_state.players[0].right_tower_hp <= 0:
                    zones.append((9, self.RIVER_Y1 - 4, self.width, self.RIVER_Y1))  # Right half: x=9-17, y=11-14
        return zones
    
    def can_deploy_at(self, pos: Position, player_id: int, battle_state=None, is_spell=False, spell_obj=None) -> bool:
        """Check if position is valid for deployment"""
        # Check basic bounds
        if not self.is_valid_position(pos) or self.is_blocked_tile(int(pos.x), int(pos.y)): return False
        if not is_spell and self.is_tower_tile(pos, battle_state): return False
        if is_spell and not self._is_rolling_projectile_spell(spell_obj): return True
        if (pos.y == 0 or pos.y == 31) and not (6 <= pos.x <= 11): return False
        zones = self.get_deploy_zones(player_id, battle_state)
        for x1, y1, x2, y2 in zones:
            if x1 <= pos.x < x2 and y1 <= pos.y < y2:
                return True
        if player_id == 0 and pos.y == 0 and 6 <= pos.x <= 11:
            return True
        elif player_id == 1 and pos.y == 31 and 6 <= pos.x <= 11:
            return True
        return False

    def get_tower_blocked_x_ranges(self, y: float, battle_state=None):
        """Get X coordinate ranges blocked by towers at a specific Y coordinate"""
        blocked_ranges = []
        for tower_pos, radius, player_id in self.towers:
            # Check if tower is still alive
            if battle_state:
                tower_alive = self._is_tower_alive(tower_pos, player_id, battle_state)
                if not tower_alive:
                    continue
            # Check if Y coordinate intersects with tower area
            dy = abs(y - tower_pos.y)
            if dy <= radius:
                # Y coordinate overlaps with tower, add X range to blocked list
                x_min = tower_pos.x - radius
                x_max = tower_pos.x + radius
                blocked_ranges.append((x_min, x_max))
        return blocked_ranges

if __name__ == '__main__':
    print(TileGrid().is_blocked_tile(int(6.9), int(16.001)))
