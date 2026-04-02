from core import BasicCharacter, Position
from card_utils import Card

class Witch(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
        self.next_spawn_remaining = 1.0
    def on_tick(self, dt):
        super().on_tick(dt)
        if not self.entity.is_alive: return
        if self.next_spawn_remaining > 0:
            self.next_spawn_remaining -= dt
            return
        # spawn skeletons!
        from battle import get_spawn_position, Troop
        skeleton = Card('Skeletons')
        skeleton.spawn_number = 4
        skeleton.spawn_radius = 2
        skeleton.spawn_delay = 0
        positions = get_spawn_position(skeleton, self.entity.position, self.entity.player, False)
        for each in positions:
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, each, self.entity.player, 'Skeletons'))
        self.next_spawn_remaining = 7.0

class Balloon(BasicCharacter):
    def on_death(self):
        from battle import TimedExplosive
        bomb = TimedExplosive(self.battle_state.next_entity_id, self.entity.position, self.entity.player, self.entity.name)
        self.battle_state._spawn_entity(bomb)

class Golem(BasicCharacter):
    def on_death(self):
        from battle import Troop, Position
        self.battle_state = self.entity.battle_state

        positions = [Position(self.entity.position.x-0.5, self.entity.position.y),
                     Position(self.entity.position.x+0.5, self.entity.position.y)]
        for position in positions:
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, position, self.entity.player, 'Golemite'))


class Prince(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
        self.starting_position = Position(self.entity.position.x, self.entity.position.y)
        self.charging = False

    def on_tick(self, dt):
        super().on_tick(dt)
        distance = self.entity.position.distance_to(self.starting_position)
        if distance > self.entity.data.charge_range and not self.charging:
            print('Prince starts to charge at', self.entity.position.x, self.entity.position.y, distance, self.entity.data.charge_range)
            print(self.entity.name, self.entity.data.charge_range)
            self.charging = True
            self.entity.speed *= 2
        if self.charging: self.entity.attack_cooldown = 0

    def on_attack(self, current_target=None):
        if not self.charging:
            current_target.take_damage(self.entity.data.damage)
            self.starting_position = Position(self.entity.position.x, self.entity.position.y)
        else:
            current_target.take_damage(self.entity.data.charge_damage)
            print('Prince stopped charging at position', self.entity.position.x, self.entity.position.y)
            self.charging = False
            self.starting_position = Position(self.entity.position.x, self.entity.position.y)
            self.entity.speed = self.entity.data.speed
        self.entity.attack_cooldown = self.entity.data.hit_speed

class GiantSkeleton(BasicCharacter):
    def on_death(self):
        from battle import TimedExplosive
        bomb = TimedExplosive(self.battle_state.next_entity_id, self.entity.position, self.entity.player,
                              self.entity.name)
        self.battle_state._spawn_entity(bomb)
