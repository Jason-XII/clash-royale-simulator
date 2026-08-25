/// <reference types="frida-gum" />
'use strict';

const SNAPSHOT_INTERVAL_MS = 100;
const OFFSETS = {
  tick: 0x8d97a0,
  getManager: 0xbb6508,
  getBattleObj: 0xbb93e0,
  getHpState: 0xbb05ac,
  getTowerHp: 0xcf719c,
  getAccountContext: 0xc85c28,
  getLocalAccountObject: 0xc8a954,
  getCardByHandSlot: 0xf37da0,
  getNextCard: 0xf3e1a8,
  cardObjectToData: 0xfef788,
  inboundDispatch: 0x1386fc8,
};

let holder = ptr(0);
let getManager, getBattleObj, getHpState, getTowerHp, getAccountContext, getLocalAccountObject;
let getCardByHandSlot, getNextCard, cardObjectToData;

function p64(address, offset) {
  // read an eight-byte pointer.
  try { return address.add(offset).readPointer(); } catch (_) { return ptr(0); }
}
function i32(address, offset) {
  // read signed 32-bit integer.
  try { return address.add(offset).readS32(); } catch (_) { return null; }
}
function u32(address, offset) {
  // read unsigned 32-bit integer.
  try { return address.add(offset).readU32(); } catch (_) { return null; }
}
function f32(address, offset) {
  // read 32-bit float.
  try { return address.add(offset).readFloat(); } catch (_) { return null; }
}

function readEntity(entity) {
  const kind = i32(entity, 0x30);
  const side = i32(entity, 0x78);
  const x = i32(entity, 0x7c);
  const y = i32(entity, 0x80);
  const x2 = i32(entity, 0x84);
  const y2 = i32(entity, 0x88);
  const cardId = i32(entity, 0xac);
  const levelIndex = i32(entity, 0x120);

  // Passive HP path verified by the x86 implementation.
  const owner = p64(entity, 0x18);
  const component = p64(owner, 0x10);
  if (owner.isNull() || component.isNull()) return null;
  const hp = i32(component, 0x10);
  const maxHp = i32(component, 0x14);

  return {
    event: 'entity_observed',
    ptr: entity.toString(),
    kind_30: kind,
    side_78: side,
    pos_x_7c: x,
    pos_y_80: y,
    pos_x2_84: x2,
    pos_y2_88: y2,
    card_id_ac: cardId,
    level_index_120: levelIndex,
    hp_10: hp,
    max_hp_14: maxHp,
    max_hp_18: i32(component, 0x18),
    source: 'battle_entity_list',
  };
}

function readBattleEntities(hpState) {
  if (hpState.isNull()) return { valid: false, entities: [] };

  // hpState+0x08 -> registry+0x40 -> collection. The collection stores its
  // pointer array at +0x08 and signed count at +0x14.
  const registry = p64(hpState, 0x08);
  const collection = p64(registry, 0x40);
  const data = p64(collection, 0x08);
  const count = i32(collection, 0x14);

  const entities = [];
  for (let index = 0; index < count; index++) {
    const entity = p64(data, index * Process.pointerSize);
    if (entity.isNull()) continue;
    const observation = readEntity(entity);
    if (observation !== null) entities.push(observation);
  }
  return { valid: true, entities };
}

function readPlayerIdentity(manager) {
  try {
    const battle = getBattleObj(manager);
    const roster = p64(battle, 0xa8);
    const accountContext = getAccountContext();
    const localAccount = getLocalAccountObject(accountContext, 0, 0);
    const count = i32(roster, 0x60);
    if (battle.isNull() || roster.isNull() || localAccount.isNull() || count < 2 || count > 6) return null;

    const localId = localAccount.add(0x20);
    const players = [];
    for (let index = 0; index < count; index++) {
      const account = p64(roster, 0x30 + index * 8);
      players.push(u32(account, 4));
    }
    return [u32(localId, 4), players];
  } catch (_) {
    return null;
  }
}

setImmediate(function () {
  const module = Process.findModuleByName('libg.so');
  const base = module.base;

  getManager = new NativeFunction(base.add(OFFSETS.getManager), 'pointer', []);
  getBattleObj = new NativeFunction(base.add(OFFSETS.getBattleObj), 'pointer', ['pointer']);
  getHpState = new NativeFunction(base.add(OFFSETS.getHpState), 'pointer', ['pointer']);
  getTowerHp = new NativeFunction(base.add(OFFSETS.getTowerHp), 'int', ['pointer', 'int']);
  getAccountContext = new NativeFunction(base.add(OFFSETS.getAccountContext), 'pointer', []);
  getLocalAccountObject = new NativeFunction(base.add(OFFSETS.getLocalAccountObject), 'pointer', ['pointer', 'int', 'int']);
  getCardByHandSlot = new NativeFunction(base.add(OFFSETS.getCardByHandSlot), 'pointer', ['pointer', 'int']);
  getNextCard = new NativeFunction(base.add(OFFSETS.getNextCard), 'pointer', ['pointer']);
  cardObjectToData = new NativeFunction(base.add(OFFSETS.cardObjectToData), 'pointer', ['pointer', 'int']);

  Interceptor.attach(base.add(OFFSETS.tick), {
    // `holder` is the current game runtime holder object.
    onEnter(args) {
      holder = args[0];
    },
  });

  Interceptor.attach(base.add(OFFSETS.inboundDispatch), {
    // intercepts message type 23877 and sends plain text to the python script
    onEnter(args) {
      if (args[1].toInt32() !== 23877) return;
      const payloadLength = args[4].toInt32();
      if (payloadLength <= 0 || payloadLength > 64 * 1024) return;
      try {
        const payload = args[3].readPointer().readByteArray(payloadLength);
        send({
          event: 'inbound_plaintext',
          t_ms: Date.now(),
          message_id: 23877,
          version: args[2].toInt32(),
          length: payloadLength,
        }, payload);
      } catch (_) {
      }
    },
  });

  setInterval(function () {
    if (holder.isNull()) return;
    const uiState = p64(holder, 0xa8);
    const handModel = p64(holder, 0x290);
    const hand = [];

    for (let slot = 0; slot < 4; slot++) {
      // gets current hand
      try {
        const cardObject = getCardByHandSlot(handModel, slot);
        const data = cardObject.isNull() ? ptr(0) : cardObjectToData(cardObject, 0);
        hand.push({
          slot,
          card_object: cardObject.toString(),
          data_pointer: data.toString(),
          deck_index_220: i32(p64(handModel, 0x220), slot * 4),
          data_id_40: i32(data, 0x40),
        });
      } catch (_) {
        hand.push({ slot, card_object: '0x0', data_pointer: '0x0', deck_index_220: null, data_id_40: null });
      }
    }

    let nextCardObject = ptr(0);
    let nextCardData = ptr(0);
    try {
      nextCardObject = getNextCard(handModel);
      nextCardData = nextCardObject.isNull() ? ptr(0) : cardObjectToData(nextCardObject, 0);
    } catch (_) {
    }

    let manager = ptr(0);
    let hpState = ptr(0);
    let upperKingHp = null;
    let lowerKingHp = null;
    try {
      manager = getManager();
      hpState = getHpState(manager);
      upperKingHp = getTowerHp(hpState, 0);
      lowerKingHp = getTowerHp(hpState, 1);
    } catch (_) {
    }

    const entitySnapshot = readBattleEntities(hpState);
    send({
      event: 'runtime_snapshot',
      t_ms: Date.now(),
      holder: holder.toString(),
      ui_state: uiState.toString(),
      hand_model: handModel.toString(),
      own_elixir_1e0: i32(uiState, 0x1e0),
      battle_clock_220: f32(uiState, 0x220),
      hand,
      next_queue_count_23c: i32(handModel, 0x23c),
      next_queue_deck_index_230: i32(p64(handModel, 0x230), 0),
      next_card_object: nextCardObject.toString(),
      next_card_data: nextCardData.toString(),
      next_card_data_id_40: i32(nextCardData, 0x40),
      manager: manager.toString(),
      hp_state: hpState.toString(),
      player_identity: readPlayerIdentity(manager),
      upper_king_hp: upperKingHp,
      lower_king_hp: lowerKingHp,
      entity_list_valid: entitySnapshot.valid,
      entities: entitySnapshot.entities,
    });
  }, SNAPSHOT_INTERVAL_MS);

  send({
    event: 'capture_ready',
    offsets: OFFSETS,
    entity_layout: {
      registry: 0x08,
      collection: 0x40,
      data: 0x08,
      count: 0x14,
      owner: 0x18,
      hp_component: 0x10,
    },
  });
});

// finally, we set a placeholder that keeps the script running and listens to activities
setInterval(function keepAlive() {}, 1000);
