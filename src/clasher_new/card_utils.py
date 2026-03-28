import json
from fastcore.all import nested_idx

with open('gamedata.json') as f:
    data = json.load(f)

with open('cards_stats_characters.json') as f:
    extra = json.load(f)

air_units = [each['name'] for each in extra if each['flying_height'] != 0]
data = data['items']['spells']
card_data = {each['name']: each for each in data}

# The king tower is not defined in `gamedata.json`, have to hard code it here.
king_tower_stats = {
    'name': 'KingTower',
    'summonCharacterData': {
        'hitpoints': 4824,
        'hitSpeed': 800,
        'damage': 109,
        'sightRange': 7000,
        'range': 7000,
        'collisionRadius': 1750,
        'tidTarget': 'TID_TARGETS_AIR_AND_GROUND',
        'deployTime': 0.0,
        'projectileData': {
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

        self.area_damage_radius = self.data['summonCharacterData'].get('areaDamageRadius', 0) / 1000
        self.projectile_damage_radius = nested_idx(self.data, 'summonCharacterData', 'projectileData', 'spawnProjectileData', 'radius')
        self.collision_radius = self.data['summonCharacterData'].get('collisionRadius', 1000) / 1000
        self.hit_speed = self.data['summonCharacterData'].get('hitSpeed') / 1000
        self.load_time = self.data['summonCharacterData'].get('loadTime', 0) / 1000
        self.speed = self.data['summonCharacterData'].get('speed', 0)/60
        self.target_only_buildings = self.data['summonCharacterData']['tidTarget'] == "TID_TARGETS_BUILDINGS"
        self.is_air_unit = self.name in air_units
        self.attack_air = 'AIR' in self.data['summonCharacterData'].get("tidTarget", '')
        self.attack_ground = ('GROUND' in self.data['summonCharacterData']['tidTarget']) or self.target_only_buildings
        self.range = self.data['summonCharacterData']['range'] / 1000
        self.sight_range = self.data['summonCharacterData']['sightRange'] / 1000
        self.deploy_time = self.data['summonCharacterData'].get('deployTime', 0) / 1000
        self.charge_range = self.data['summonCharacterData'].get('chargeRange')
        self.projectiles = 'projectileData' in self.data['summonCharacterData']
        self.projectile_data = Projectile(self.data['summonCharacterData'].get('projectileData', {}))
        self.area_effect_data = AreaEffectData(self.data.get('areaEffectObjectData', {}))
        self.charge_damage = self.data['summonCharacterData'].get('damageSpecial', 0)

        self.lifetime = self.data['summonCharacterData'].get('lifeTime', float('inf'))

        if self.name == 'King_PrincessTowers':
            self.collision_radius = 1.25

class Projectile:
    def __init__(self, projectile_data):
        self.data = projectile_data
        self.damage = self.data.get('damage')
        self.speed = self.data.get('speed', 0) / 60
        self.radius = self.data.get('spawnProjectileData', {}).get('radius', 0)
        self.target_buff = self.data.get('targetBuffData', {})
        self.buff_time = self.data.get('buffTime', 0) / 1000
        self.hits_air = 'AIR' in self.data.get('tidTarget', '')
        self.hits_ground = 'GROUND' in self.data.get('tidTarget', '') or 'BUILDING' in self.data.get('tidTarget', '')
        self.pushback = self.data.get('pushback', 0)
        self.name = self.data.get('name', 'Unknown')
        if self.data.get('name') == 'TowerPrincessProjectile':
            self.hits_air = True
            self.hits_ground = True

class AreaEffectData:
    def __init__(self, area_effect_data):
        self.data = area_effect_data
        self.duration = self.data.get('lifeDuration', 0) / 1000
        self.buff_data = self.data.get('buffData', {})
        self.speed_multiplier = self.buff_data.get('speedMultiplier')
        self.crown_tower_damage_percent = self.buff_data.get('crown', 0) or self.data.get('crownTowerDamagePercent', 0)




if __name__ == '__main__':
    print(air_units)