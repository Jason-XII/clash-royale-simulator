class BaseMechanic:
    def on_attach(self, entity): pass
    def on_spawn(self, entity): pass
    def on_tick(self, entity, dt_ms: int): pass
    def on_attack_start(self, entity, target): pass
    def on_attack_hit(self, entity, target): pass
    def on_death(self, entity): pass