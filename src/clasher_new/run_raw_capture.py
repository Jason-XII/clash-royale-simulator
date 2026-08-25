from pathlib import Path
import time
import subprocess

import frida

def adb(*args):
    return subprocess.run(
        ["adb", *args],
        check=True,
        text=True,
        capture_output=True,
    )

PACKAGE = "nullsroyale.rel.free"
FRIDA_SERVER = "/data/local/tmp/frida-server"
SECONDS = 300
current_dir = Path('.')

source = (current_dir / "hook_raw_capture.js").read_text()

# ensure the frida server has started
adb("forward", "tcp:27042", "tcp:27042")
adb("shell", "chmod", "755", FRIDA_SERVER)
adb("shell", "sh", "-c", f"{FRIDA_SERVER} >/dev/null 2>&1 &")
device = frida.get_device_manager().add_remote_device("127.0.0.1:27042")
result = adb("shell", "pidof", PACKAGE)
pids = result.stdout.strip().split()
if not pids:
    raise SystemExit(f"{PACKAGE} is not running")
pid = int(pids[0])

session = device.attach(pid)
sequence = 0
print('【应用程序】成功连接到安卓模拟器，检测到Nulls Royale正常运行。')

entities = {}
local_arena_side = None
player_identity = None

def mainloop(visualizer=None):
    def on_message(message, data):
        global sequence, local_arena_side, player_identity
        if message["type"] != "send":
            print(message)
            return
        payload = message["payload"]
        if payload['event'] == 'runtime_snapshot':
            if visualizer is not None:
                visualizer.snapshot = payload
                identity = payload.get('player_identity')
                if isinstance(identity, list) and len(identity) == 2:
                    local_id, id_lst = identity
                    if isinstance(id_lst, list) and local_id in id_lst:
                        visualizer.local_player_index = id_lst.index(local_id)
        if visualizer is not None: visualizer.entities = entities
        else:
            print(payload)

    script = session.create_script(source)
    script.on("message", on_message)
    script.load()

    try:
        time.sleep(SECONDS)
    except KeyboardInterrupt:
        pass

    session.detach()

if __name__ == '__main__':
    mainloop()
