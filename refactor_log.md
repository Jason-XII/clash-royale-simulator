# Refinements made:

## Current Progress

- `arena.py` finished. 267 → 156 lines
- `player.py` finished. 78 → 37 lines
- created `card_utils.py`
- deleted the `factory` directory and `data.py`
- completely refactored `entities.py`'s `Entity`, `Troop`, `Building` base class, from 1030 lines to 380 lines.

## Caveats

- The `gamedata.json` files is not describing the behavior of firecracker correctly.
- May cause exceptions in `entities.py` because I didn't handle cards that doesn't have some specific attributes.
- Removed the "temporary buff" judgement in `entities.py`. May cause problems.
- Targets troops more than buildings? Is that true?
- targeting uses `sorted` to sort by distance, which may cause problems.
- "Apply projectile knockback" may be buggy if the result tile is not walkable.

## card_factory.py
Fixed possible bug in `_determine_card_kind`: 
champions was previously determined as troops, but now I compressed the logic to only two lines.

There is no such thing as `summonSpellData` in any entries. I removed it, but will take notice of how spells do their targeting. Is `_create_targeting_behaviour` only used for troops?

The program defines classes within functions, which is a weird way of coding. I put the classes outside. And the classes look a lot like placeholders that don't do anything.

The card factory logic is weird. May have to change it later.

# Redoing the entire project

## 1. Examine how the card data are loaded

I deleted `card_data.py` and `card_aliases.py`. They mainly contain protocols that complicate the program structure, so I removed them.
Maybe later if they prove to be useful, I'll add them back.

## 2. Massively reconfigured `arena.py`

`arena.py` contains only two parts of important information:
- the `Position` class which is just like namedtuple;
- `TileGrid` class which has defined some important rules on where to and when you can deploy troops.

## 3. Deleted unnecessary files entirely

Engine.py requires `battle.py` so I'll move that for later.
I refactored `player.py` and reduced code length to half its original size.

Also created `card_utils.py` to get necessary information of a card when needed. I don't want long prototypes.

## 4. Modified the start of `entities.py`

Needs to check `mechanics` further before continuing. 

It seems that `mechanics` also heavily relies on `entities.py`.