from core import BasicCharacter, Position
from card_utils import Card
from arena import TileGrid

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

class LavaHound(BasicCharacter):
    def on_death(self):
        from battle import get_spawn_position, Troop
        self.battle_state = self.entity.battle_state
        positions = get_spawn_position(Card('LavaPups'), self.entity.position, self.entity.player)
        for position in positions:
            self.battle_state._spawn_entity(Troop(self.battle_state.next_entity_id, position, self.entity.player, 'LavaPups'))

class Prince(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
        self.starting_position = Position(self.entity.position.x, self.entity.position.y)
        self.charging = False

    def on_tick(self, dt):
        super().on_tick(dt)
        distance = self.entity.position.distance_to(self.starting_position)
        if distance > self.entity.data.charge_range and not self.charging:
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
            self.charging = False
            self.starting_position = Position(self.entity.position.x, self.entity.position.y)
            self.entity.speed = self.entity.data.speed
        self.entity.attack_cooldown = self.entity.data.hit_speed

class DarkPrince(Prince):
    pass

class GiantSkeleton(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
    def on_death(self):
        from battle import TimedExplosive
        bomb = TimedExplosive(self.battle_state.next_entity_id, self.entity.position, self.entity.player,
                              self.entity.name)
        self.battle_state._spawn_entity(bomb)

class IceWizard(BasicCharacter):
    def on_spawn(self):
        spawn_data = self.entity.data.spawn_data
        for entity in self.entity.battle_state.entities.values():
            if not entity.is_alive or entity.player == self.entity.player: continue
            if not entity.position.distance_to(self.entity.position) < spawn_data['radius']/1000 + entity.data.collision_radius:
                continue
            entity.take_damage(spawn_data['damage'])
            entity.speed_debuff = min(1 + spawn_data['buffData']['speedMultiplier'] / 100, entity.speed_debuff)
            entity.debuff_time_remaining = spawn_data['buffTime']/1000

class Miner(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
        self.distance = self.entity.position.distance_to(TileGrid.RED_KING_TOWER if self.entity.player == 1 else TileGrid.BLUE_KING_TOWER)
        self.freeze_time = self.distance/(650/60)
        self.entity.targetable = False
        self.entity.invincible = True

    def on_tick(self, dt):
        if self.freeze_time > 0:
            self.freeze_time -= dt
            self.entity.deploy_delay_remaining = self.entity.data.deploy_time
        else:
            self.entity.targetable = True
            self.entity.invincible = False

class Rage(BasicCharacter):
    def __init__(self, entity):
        super().__init__(entity)
        self.deploy_delay_remaining = entity.data.deploy_time
        self.data = entity.data.death_area_effect
        self.lifetime = self.data['lifeDuration']/1000

        self.radius = self.data['radius']/1000
        self.hit_speed = self.data['hitSpeed']/1000
        self.buff_time = self.data['buffTime']/1000
        self.speed_multiplier = self.data['buffData']['hitSpeedMultiplier']/100
        self.damage = self.data['spawnAreaEffectObjectData']['damage']
        self.crown_percent = self.data['spawnAreaEffectObjectData']['crownTowerDamagePercent']/100 + 1
        self.attack_cooldown = 0
        self.entity.battle_state.deal_area_damage(self.entity.player, self.entity.position, self.radius, self.damage,
                                                  True, True,
                                                  self.crown_percent)

    def on_tick(self, dt):
        super().on_tick(dt)
        from battle import Troop, Building, Projectile
        # print(self.attack_cooldown)
        if self.deploy_delay_remaining > 0:
            self.deploy_delay_remaining -= dt
            return
        if self.lifetime <= 0:
            self.entity.is_alive = False
            return
        else:
            self.lifetime -= dt
        if self.attack_cooldown <= 0:
            for entity in self.entity.battle_state.entities.values():
                if not entity.is_alive or entity.player != self.entity.player: continue
                if entity.position.distance_to(self.entity.position) > self.radius + entity.data.collision_radius: continue
                if isinstance(entity, Troop) or isinstance(entity, Building):
                    print('Adding buff to, ', entity.name)
                    entity.speed_buff = max(entity.speed_buff, self.speed_multiplier)
                    entity.buff_time_remaining = max(entity.buff_time_remaining, self.buff_time)

            self.attack_cooldown = self.hit_speed
            pass
        else:
            self.attack_cooldown -= dt

class RageBarbarian(BasicCharacter):
    def on_death(self):
        from battle import Entity
        self.battle_state._spawn_entity(Entity(self.battle_state.next_entity_id, self.entity.position, self.entity.player, "Rage", self.battle_state))