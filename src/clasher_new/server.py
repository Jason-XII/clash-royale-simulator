import socket, json, threading, time
from battle import BattleState, Building
from player import PlayerState
from arena import Position

TICK_RATE = 60
DT = 1.0 / TICK_RATE

class GameServer:
    def __init__(self, host='10.235.130.132', port=9999):
        self.host, self.port = host, port
        self.clients = []      # list of (conn, player_id)
        self.inputs = [[], []] # pending inputs per player
        self.lock = threading.Lock()
        self.battle = None
        self.decks = [None, None]

    def send(self, conn, msg):
        data = (json.dumps(msg) + '\n').encode()
        conn.sendall(data)

    def broadcast(self, msg):
        for conn, _ in self.clients:
            try: self.send(conn, msg)
            except: pass

    def get_state(self):
        return {
            'time': self.battle.time,
            'game_over': self.battle.game_over,
            'winner': self.battle.winner,
            'elixir': [p.elixir for p in self.battle.players],
            'entities': [
                e.to_dict()
                for e in self.battle.entities.values() if e.is_alive
            ]
        }

    def handle_client(self, conn, player_id):
        buf = ''
        while True:
            try:
                buf += conn.recv(4096).decode()
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    msg = json.loads(line)
                    if msg['type'] == 'deck':
                        self.decks[player_id] = msg['cards']
                        continue
                    with self.lock:
                        self.inputs[player_id].append(msg)
            except socket.timeout:
                continue
            except:
                break

    def run(self):
        # Wait for 2 players
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((self.host, self.port))
        srv.listen(2)
        print(f"Waiting for 2 players on {self.host}:{self.port}...")
        while len(self.clients) < 2:
            conn, addr = srv.accept()
            conn.settimeout(0.01)
            pid = len(self.clients)
            self.clients.append((conn, pid))
            self.send(conn, {'type': 'hello', 'player_id': pid})
            threading.Thread(target=self.handle_client, args=(conn, pid), daemon=True).start()
            print(f"Player {pid} connected from {addr}")

        # Start game
        while not all(self.decks):
            time.sleep(0.05)

        self.battle = BattleState(
            PlayerState(0, self.decks[0], 5),
            PlayerState(1, self.decks[1], 5)
        )
        self.battle = BattleState(PlayerState(0, self.decks[0], 5), PlayerState(1, self.decks[1], 5))
        self.broadcast({'type': 'start'})
        print("Game started!")

        # Main loop
        while not self.battle.game_over:
            t0 = time.perf_counter()
            with self.lock:
                for pid, input_list in enumerate(self.inputs):
                    for inp in input_list:
                        if inp['type'] == 'deploy':
                            self.battle.deploy_card(pid, inp['card'], Position(inp['x'], inp['y']))
                    self.inputs[pid] = []
            self.battle.step(DT)
            self.broadcast({'type': 'state', **self.get_state()})
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, DT - elapsed))

        self.broadcast({'type': 'state', **self.get_state()})
        print(f"Game over! Winner: Player {self.battle.winner}")

if __name__ == '__main__':
    GameServer().run()
