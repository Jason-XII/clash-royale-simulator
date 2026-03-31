from jedi.api.helpers import get_stack_at_position

from .card_utils import Card

class BasicCharacter:
    def __init__(self, entity):
        self.entity = entity
        self.battle_state = self.entity.battle_state
        self.data = self.entity.data
    def on_tick(self, dt): self.battle_state = self.entity.battle_state
    def on_death(self): pass

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
        from .battle import get_spawn_position, Troop
        skeleton = Card('Skeletons')
        skeleton.spawn_number = 4
        skeleton.spawn_radius = 2
        skeleton.spawn_delay = 0
        positions = get_spawn_position(skeleton, self.entity.position, self.entity.player, False)
        for each in positions:
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, each, self.entity.player, 'Skeletons'))
        self.next_spawn_remaining = 7.0

class Balloon(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
    def on_death(self):
        from .battle import TimedExplosive
        bomb = TimedExplosive(self.battle_state.next_entity_id, self.entity.position, self.entity.player, self.entity.name)
        self.battle_state._spawn_entity(bomb)

class Golem(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
    def on_death(self):
        from .battle import Troop, Position
        self.battle_state = self.entity.battle_state

        positions = [Position(self.entity.position.x-0.5, self.entity.position.y),
                     Position(self.entity.position.x+0.5, self.entity.position.y)]
        for position in positions:
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, position, self.entity.player, 'Golemite'))

