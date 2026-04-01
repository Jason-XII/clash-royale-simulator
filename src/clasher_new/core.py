from dataclasses import dataclass
import math

@dataclass
class Position:
    x: float
    y: float
    def distance_to(self, other):
        return math.hypot(self.x-other.x, self.y-other.y)

class BasicCharacter:
    def __init__(self, entity):
        self.entity = entity
        self.battle_state = self.entity.battle_state
        self.data = self.entity.data
    def on_tick(self, dt): pass
    def on_death(self): pass
    def on_attack(self, current_target=None):
        if self.entity.data.damage:
            if self.entity.data.area_damage_radius:
                self.battle_state.deal_area_damage(self.entity.player, self.entity.position, self.data.area_damage_radius,
                                                   self.data.damage,
                                                   self.data.attack_air, self.data.attack_ground)
            else:
                current_target.take_damage(self.data.damage)
        elif self.entity.data.projectiles:
            # must have projectiles
            self.entity.create_projectile(current_target)
        self.entity.attack_cooldown = self.data.hit_speed