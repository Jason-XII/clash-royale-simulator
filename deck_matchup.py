#!/usr/bin/env python3
import sys, random
sys.path.append('src')

from random_battle import RandomBattleSimulator

class DeckMatchup(RandomBattleSimulator):
    def assign_random_decks_to_players(self):
        decks = {
            0: {"name": "Royal Recruits & Royal Hogs",
                "cards": ["RoyalRecruits", "RoyalHogs", "DartBarrell", "Fireball",
                          "Zap", "GoblinCage", "BarbarianBarrel", "Heal"]},
            1: {"name": "Log Bait",
                "cards": ["GoblinBarrel", "GoblinGang", "IceSpirit",
                          "InfernoTower", "Knight", "Princess", "Rocket", "Log"]},
        }
        self.player_cycles = {0: [], 1: []}
        self.player_cycle_indices = {0: 0, 1: 0}
        self._chosen_deck_names = {0: "", 1: ""}

        for pid, player in enumerate(self.battle.players):
            chosen = decks[pid]
            cycle = chosen["cards"][:]
            random.shuffle(cycle)
            self.player_cycles[pid] = cycle
            self.player_cycle_indices[pid] = 0
            player.deck = chosen["cards"][:]
            player.hand = cycle[:4]
            self._chosen_deck_names[pid] = chosen["name"]
            print(f"[Deck] Player {pid}: {self._chosen_deck_names[pid]} -> {', '.join(cycle)}")

if __name__ == "__main__":
    sim = DeckMatchup()
    sim.run()
