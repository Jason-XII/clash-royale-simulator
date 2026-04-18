"""bench_batched_mp.py — Multiprocess batched self-play benchmark"""
import time, torch, numpy as np
import torch.multiprocessing as mp
from env import SelfPlayEnv
from env_model import ClashAgent

DECK_A = ["Knight", "Archer", "Giant", "Musketeer",
           "Valkyrie", "Bomber", "Prince", "BabyDragon"]
DECK_B = ["Pekka", "MiniPekka", "Wizard", "Goblins",
           "HogRider", "Witch", "Barbarians", "Minions"]

def worker(worker_id, obs_queue, action_queue):
    """Each worker owns one env, loops: step → send obs → recv action"""
    env = SelfPlayEnv(DECK_A, DECK_B)
    obs0, obs1, _ = env.reset()
    obs_queue.put((worker_id, obs0, obs1, False, {}))
    while True:
        a0, a1 = action_queue.get()
        if a0 is None: break   # shutdown signal
        obs0, obs1, r0, r1, done, info = env.step(a0, a1)
        if done:
            obs0, obs1, _ = env.reset()
        obs_queue.put((worker_id, obs0, obs1, done, info))

def stack(obss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "arena":  torch.tensor(np.stack([o["arena"]  for o in obss]), device=device),
        "global": torch.tensor(np.stack([o["global"] for o in obss]), device=device),
        "hand":   torch.tensor(np.stack([o["hand"]   for o in obss]), device=device),
    }

if __name__ == "__main__":
    B = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Workers: {B}")

    model = ClashAgent().to(device)
    model.eval()
    hx0 = model.init_hidden(B, device)
    hx1 = model.init_hidden(B, device)

    obs_queue    = mp.Queue()
    action_queues = [mp.Queue() for _ in range(B)]

    procs = [mp.Process(target=worker, args=(i, obs_queue, action_queues[i]),
                        daemon=True) for i in range(B)]
    for p in procs: p.start()

    # Collect initial obs from all workers
    obs0s, obs1s = [None]*B, [None]*B
    for _ in range(B):
        wid, o0, o1, _, _ = obs_queue.get()
        obs0s[wid], obs1s[wid] = o0, o1

    games_done, t0 = 0, time.perf_counter()
    with torch.no_grad():
        while games_done < 200:
            a0, _, _, hx0 = model.get_action(stack(obs0s), hx0)
            a1, _, _, hx1 = model.get_action(stack(obs1s), hx1)

            for i in range(B):
                act0 = {"card": int(a0["card"][i]), "position": a0["position"][i]}
                act1 = {"card": int(a1["card"][i]), "position": a1["position"][i]}
                action_queues[i].put((act0, act1))

            for _ in range(B):
                wid, o0, o1, done, info = obs_queue.get()
                obs0s[wid], obs1s[wid] = o0, o1
                if done:
                    games_done += 1
                    hx0[0][wid].zero_(); hx0[1][wid].zero_()
                    hx1[0][wid].zero_(); hx1[1][wid].zero_()

    elapsed = time.perf_counter() - t0
    print(f"200 games in {elapsed:.1f}s  →  {3600/(elapsed/200):.0f} games/hr")

    for q in action_queues: q.put((None, None))
    for p in procs: p.join()
