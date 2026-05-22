import json
from fastcore.all import nested_idx

with open('gamedata.json') as f:
    data = json.load(f)

with open('cards_stats_characters.json') as f:
    characters_data = json.load(f)
    air_units = [each['name'] for each in characters_data if each['flying_height'] != 0]
    characters = {each['name']:each for each in characters_data}
with open('cards_stats_spell.json') as f:
    spells_data = json.load(f)
    spells = {each['name']:each for each in spells_data}
with open('cards_stats_building.json') as f:
    buildings_data = json.load(f)
    buildings = {each['name']:each for each in buildings_data}
with open('cards_stats_projectile.json') as f:
    projectiles = {each['name']:each for each in json.load(f)}

data = data['items']['spells']
card_data = {each['name']: each for each in data}

card_data['Golemite'] = {'name': 'Golemite', 'summonCharacterData':card_data['Golem']['summonCharacterData']['deathSpawnCharacterData']}

lava_pups = card_data['LavaHound']['summonCharacterData']['deathSpawnCharacterData']
barbarian = card_data['BattleRam']['summonCharacterData']['deathSpawnCharacterData']
card_data['LavaPups'] = {'name': 'LavaPups', 'summonCharacterData':lava_pups} | lava_pups
card_data['Barbarian'] = {'name': 'Barbarian', 'summonCharacterData': barbarian} | barbarian

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
        self.data.setdefault('summonCharacterData', self.data)
        self.hp = self.data['summonCharacterData'].get('hitpoints', 0)
        self.elixir = self.data.get('manaCost') # princess towers don't have elixir cost
        self.name = self.data['name']
        self.damage = self.data['summonCharacterData'].get('damage', 0)
        self.spawn_number = self.data.get('summonNumber', 1)
        self.spawn_delay = self.data.get('summonDeployDelay', 0) / 1000
        self.spawn_radius = self.data.get('summonRadius', 550) / 1000

        self.area_damage_radius = self.data['summonCharacterData'].get('areaDamageRadius', 0) / 1000
        self.projectile_damage_radius = nested_idx(self.data, 'summonCharacterData', 'projectileData', 'spawnProjectileData', 'radius')
        self.collision_radius = self.data['summonCharacterData'].get('collisionRadius', 1000) / 1000
        self.hit_speed = self.data['summonCharacterData'].get('hitSpeed', 0) / 1000
        self.load_time = self.data['summonCharacterData'].get('loadTime', 0) / 1000
        self.speed = self.data['summonCharacterData'].get('speed', 0)/60
        self.target_only_buildings = self.data['summonCharacterData'].get('tidTarget', '') == "TID_TARGETS_BUILDINGS"
        self.is_air_unit = self.name in air_units or self.data['summonCharacterData'].get('name', '') in air_units
        self.attack_air = 'AIR' in self.data['summonCharacterData'].get("tidTarget", '')
        self.attack_ground = ('GROUND' in self.data['summonCharacterData'].get('tidTarget', '')) or self.target_only_buildings
        self.range = self.data['summonCharacterData'].get('range', 0) / 1000
        self.sight_range = self.data['summonCharacterData'].get('sightRange', 0) / 1000
        self.deploy_time = self.data['summonCharacterData'].get('deployTime', 0) / 1000
        self.charge_range = self.data['summonCharacterData'].get('chargeRange', 0) / 1000
        self.projectiles = 'projectileData' in self.data['summonCharacterData']
        self.projectile_data = Projectile(self.data['summonCharacterData'].get('projectileData', {}))
        self.charge_damage = self.data['summonCharacterData'].get('damageSpecial', 0)
        self.shield_health = self.data['summonCharacterData'].get('shieldHitpoints', 0)

        self.lifetime = self.data['summonCharacterData'].get('lifeTime', float('inf'))

        self.death_spawn_data = self.data['summonCharacterData'].get('deathSpawnCharacterData', {})
        self.death_area_effect = self.data['summonCharacterData'].get('deathAreaEffectData', {})
        self.death_damage = self.data['summonCharacterData'].get('deathDamage', 0)

        self.jump_height = self.data['summonCharacterData'].get('jumpHeight', 0)
        self.jump_speed = self.data['summonCharacterData'].get('jumpSpeed', 0) / 60

        self.spawn_data = self.data['summonCharacterData'].get("spawnAreaObjectData", {})
        self.kamikaze = self.data['summonCharacterData'].get('kamikaze', False)

        self.tower_damage_mult = 1+self.data['summonCharacterData'].get('crownTowerDamagePercent', 0)/100

        self.type = self.data.get('tidType', '').split('_')[-1].lower()
        self.rarity = self.data.get('rarity', 'Common')

        if self.name == 'King_PrincessTowers':
            self.collision_radius = 1.5

    def set_level(self, level):
        if self.rarity == 'Common': level_index = level - 1
        elif self.rarity == 'Rare': level_index = level - 3
        elif self.rarity == 'Epic': level_index = level - 6
        elif self.rarity == 'Legendary': level_index = level - 9
        elif self.rarity == 'Champion': level_index = level - 11

        if self.projectiles:
            projectile_name = self.data['summonCharacterData']['projectileData']['name']
            self.projectile_data.damage = projectiles[projectile_name]['damage_per_level'][level_index]

        if self.type == 'character':
            character_name = self.data['summonCharacterData']['name']
            self.hp = characters[character_name]["hitpoints_per_level"][level_index]
            if self.damage:
                self.damage = characters[character_name]["damage_per_level"][level_index]
        elif self.type == 'spell':
            # For simplicity, just assume that spells are projectiles, which is already handled
            pass
        return level

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
    deck = ['Archer', 'Giant', 'Musketeer', 'MiniPekka', 'Fireball', 'Arrows', 'Minions', 'Knight']
    for each in deck:
        Card(each).set_level(11)