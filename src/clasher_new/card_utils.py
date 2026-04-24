import json
from fastcore.all import nested_idx

with open('gamedata.json') as f:
    data = json.load(f)

with open('cards_stats_characters.json') as f:
    extra = json.load(f)

air_units = [each['name'] for each in extra if each['flying_height'] != 0]
data = data['items']['spells']
card_data = {each['name']: each for each in data}

card_data['Golemite'] = {'name': 'Golemite', 'summonCharacterData':card_data['Golem']['summonCharacterData']['deathSpawnCharacterData']}

lava_pups = card_data['LavaHound']['summonCharacterData']['deathSpawnCharacterData']
card_data['LavaPups'] = {'name': 'LavaPups', 'summonCharacterData':lava_pups} | lava_pups

# The king tower is not defined in `gamedata.json`, have to hard code it here.
king_tower_stats = {
    'name': 'KingTower',
    'summonCharacterData': {
        'hitpoints': 2100,
        'hitSpeed': 1000,
        'damage': 109,
        'sightRange': 7000,
        'range': 7000,
        'collisionRadius': 1750,
        'tidTarget': 'TID_TARGETS_AIR_AND_GROUND',
        'deployTime': 3300,
        'loadTime': 700,
        'projectileData': {
            'name': 'KingTowerCannonBall',
            'speed': 600,
            'damage': 109,
        }
    }
}
card_data['KingTower'] = king_tower_stats
card_data['King_PrincessTowers']['summonCharacterData'] = card_data['King_PrincessTowers']['statCharacterData']

class Card:
    def __init__(self, card_name):
        self.data = card_data[card_name]
        self.hp = self.data['summonCharacterData']['hitpoints']
        self.elixir = self.data.get('manaCost') # princess towers don't have elixir cost
        self.name = self.data['name']
        self.damage = self.data['summonCharacterData'].get('damage', 0)
        self.spawn_number = self.data.get('summonNumber', 1)
        self.spawn_delay = self.data.get('summonDeployDelay', 0) / 1000
        self.spawn_radius = self.data.get('summonRadius', 550) / 1000

        self.area_damage_radius = self.data['summonCharacterData'].get('areaDamageRadius', 0) / 1000
        self.projectile_damage_radius = nested_idx(self.data, 'summonCharacterData', 'projectileData', 'spawnProjectileData', 'radius')
        self.collision_radius = self.data['summonCharacterData'].get('collisionRadius', 1000) / 1000
        self.hit_speed = self.data['summonCharacterData'].get('hitSpeed') / 1000
        self.load_time = self.data['summonCharacterData'].get('loadTime', 0) / 1000
        self.speed = self.data['summonCharacterData'].get('speed', 0)/60
        self.target_only_buildings = self.data['summonCharacterData']['tidTarget'] == "TID_TARGETS_BUILDINGS"
        self.is_air_unit = self.name in air_units or self.data['summonCharacterData'].get('name', '') in air_units
        self.attack_air = 'AIR' in self.data['summonCharacterData'].get("tidTarget", '')
        self.attack_ground = ('GROUND' in self.data['summonCharacterData']['tidTarget']) or self.target_only_buildings
        self.range = self.data['summonCharacterData']['range'] / 1000
        self.sight_range = self.data['summonCharacterData']['sightRange'] / 1000
        self.deploy_time = self.data['summonCharacterData'].get('deployTime', 0) / 1000
        self.charge_range = self.data['summonCharacterData'].get('chargeRange', 0) / 1000
        self.projectiles = 'projectileData' in self.data['summonCharacterData']
        self.projectile_data = Projectile(self.data['summonCharacterData'].get('projectileData', {}))
        self.area_effect_data = AreaEffectData(self.data.get('areaEffectObjectData', {}))
        self.charge_damage = self.data['summonCharacterData'].get('damageSpecial', 0)
        self.shield_health = self.data['summonCharacterData'].get('shieldHitpoints', 0)

        self.lifetime = self.data['summonCharacterData'].get('lifeTime', float('inf'))

        self.death_spawn_data = self.data['summonCharacterData'].get('deathSpawnCharacterData', {})
        self.death_damage = self.data['summonCharacterData'].get('deathDamage', 0)

        self.jump_height = self.data['summonCharacterData'].get('jumpHeight', 0)
        self.jump_speed = self.data['summonCharacterData'].get('jumpSpeed', 0) / 60

        self.spawn_data = self.data['summonCharacterData'].get("spawnAreaObjectData", {})
        self.kamikaze = self.data['summonCharacterData'].get('kamikaze', False)

        self.tower_damage_mult = 1+self.data['summonCharacterData'].get('crownTowerDamagePercent', 0)/100

        if self.name == 'King_PrincessTowers':
            self.collision_radius = 1.5

class Projectile:
    def __init__(self, projectile_data):
        self.data = projectile_data
        self.damage = self.data.get('damage')
        self.speed = self.data.get('speed', 0) / 60
        self.radius = self.data.get('spawnProjectileData', {}).get('radius', 0) or self.data.get('radius', 0) / 1000
        self.target_buff = self.data.get('targetBuffData', {})
        self.buff_time = self.data.get('buffTime', 0) / 1000
        self.hits_air = 'AIR' in self.data.get('tidTarget', '')
        self.hits_ground = 'GROUND' in self.data.get('tidTarget', '') or 'BUILDING' in self.data.get('tidTarget', '')
        self.pushback = self.data.get('pushback', 0) / 1000
        self.name = self.data.get('name', 'Unknown')
        self.roll_range = self.data.get('projectileRange', 0) / 1000
        if self.data.get('name') == 'TowerPrincessProjectile':
            self.hits_air = True
            self.hits_ground = True

class TimedExplosiveData:
    def __init__(self, death_spawn_data):
        self.data = death_spawn_data
        self.name = self.data['name']
        self.damage = self.data['deathDamage']
        self.deploy_time = self.data['deployTime'] / 1000
        self.collision_radius = self.data['collisionRadius'] / 1000
        self.range = 3.0
        self.crown_tower_damage_percent = self.data.get('crownTowerDamagePercent', 100) / 100

class AreaEffectData:
    def __init__(self, source_card_name):
        # This only works for lumberjack, will modify later.
        self.data = Card(source_card_name)['summonCharacterData'].get('deathSpawnCharacterData', {}).get('deathAreaEffectData', {})
        self.duration = self.data.get('lifeDuration', 0) / 1000
        self.radius = self.data.get('radius', 0) / 1000
        self.buff_time = self.data.get('buffTime', 0)
        self.buff_data = self.data.get('buffData', {})
        self.speed_multiplier = self.buff_data.get('speedMultiplier')
        self.damage = self.data.get('spawnAreaEffectObjectData', {}).get('damage', 0)
        self.crown_tower_damage_percent = self.buff_data.get('crown', 0) or self.data.get('crownTowerDamagePercent', 0)

if __name__ == '__main__':
    print(Card('LavaPups').spawn_number)