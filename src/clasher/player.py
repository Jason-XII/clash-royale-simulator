from card_utils import Card

# Hardcoded: max elixir is 10

class PlayerState:
    def __init__(self, player_id, cycle_queue, elixir, tower_hps=(4824, 3052, 3052)):
        self.player_id = player_id
        self.cycle = cycle_queue
        self.elixir = elixir
        self.king_tower_hp, self.left_tower_hp, self.right_tower_hp = tower_hps
    
    def regenerate_elixir(self, dt: float, base_regen_time: float = 2.8):
        elixir_per_second = 1.0 / base_regen_time
        self.elixir = min(10, self.elixir + elixir_per_second * dt)
    
    def can_play_card(self, card_name):
        return (card_name in self.cycle[:4] and
                self.elixir >= Card(card_name).elixir and
                self.king_tower_hp > 0)
    
    def play_card(self, card_name):
        """Update the player's deck when playing a card."""
        if not self.can_play_card(card_name): return False
        self.elixir -= Card(card_name).elixir
        self.cycle.remove(card_name)
        self.cycle.append(card_name)
        return True

    def get_next_card(self):
        """Return the next card in cycle, if known."""
        return self.cycle[4]
    
    def get_crown_count(self) -> int:
        """Get number of crowns (destroyed towers)"""
        if self.king_tower_hp <= 0:
            return 3
        return int(self.left_tower_hp <= 0) + int(self.right_tower_hp <= 0)
