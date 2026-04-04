import httpx, json
from pathlib import Path

Path("images").mkdir(exist_ok=True)

with open("cards.json") as f:
    cards = json.load(f)["items"]

for card in cards[:25]:
    url = card["iconUrls"]["medium"]
    name = card["name"]
    img = httpx.get(url).content
    Path(f"images/{name}.png").write_bytes(img)
    print(f"Downloaded {name}")
