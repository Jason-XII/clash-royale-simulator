from .entities import Building, Entity
from .arena import TileGrid
from .player import PlayerState

class BattleState:
    def __init__(self, player_0: PlayerState, player_1: PlayerState):
        self.entities = {}
        self.players = [player_0, player_1]
        self.arena = TileGrid()
        self.time = 0.0
        self.tick = 0
        self.dt = 1 / 30  # 33ms per tick (~30 FPS)
        self.game_over = False
        self.winner = None
        self.next_entity_id = 1
        self.regen = 2.8

        self._spawn_entity(Building(1, self.arena.RED_LEFT_TOWER, 1, 'King_PrincessTowers', []))
        self._spawn_entity(Building(2, self.arena.RED_RIGHT_TOWER, 1, 'King_PrincessTowers', []))
        self._spawn_entity(Building(3, self.arena.BLUE_LEFT_TOWER, 0, 'King_PrincessTowers', []))
        self._spawn_entity(Building(4, self.arena.BLUE_RIGHT_TOWER, 0, 'King_PrincessTowers', []))
        self._spawn_entity(Building(5, self.arena.RED_KING_TOWER, 1, 'KingTower', []))
        self._spawn_entity(Building(6, self.arena.BLUE_KING_TOWER, 0, 'KingTower', []))

    def _spawn_entity(self, entity):
        entity.battle_state = self
        self.entities[len(self.entities)+1] = entity
        entity.on_spawn()

    def tick(self, dt):
        for each in self.players:
            each.regenerate_elixir(dt, 2.8 if self.time < 120 else 1.4 if self.time < 240 else 2.8/3)
        for entity in self.entities:
            entity.update(dt, self)

    def deploy_card(self, player_id, card_name, position):
        if not self.players[player_id].can_play_card(card_name):
            return False
        self._spawn_entity(Entity(len(self.entities)+1, position, player_id, card_name, []))
        return True


