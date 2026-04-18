"""model.py — Actor-Critic agent for Clash Royale"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env import NUM_CARD_IDS, NUM_TILE_FEATURES, ARENA_H, ARENA_W


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ClashAgent(nn.Module):
    """
    Architecture:
        Arena (16ch, 32, 18)
          → replace card_id channel with learned embedding
          → CNN (ResBlocks) → spatial features (flat)
        Hand (5 ints) → card embeddings → MLP
        Global (10)   → MLP
        All concat     → LSTM → hidden
          → card logits (5)
          → per-card position heatmaps (4, 32, 18)
          → value scalar
    """

    def __init__(self, embed_dim=32, cnn_ch=64, n_res=3, lstm_hidden=256,
                 mlp_hidden=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden

        # ── Shared card embedding ────────────────────────────────
        self.card_embed = nn.Embedding(NUM_CARD_IDS, embed_dim, padding_idx=0)

        # ── Spatial encoder ──────────────────────────────────────
        # Input channels: original 16 - 1 (card_id) + embed_dim
        spatial_in = NUM_TILE_FEATURES - 1 + embed_dim
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(spatial_in, cnn_ch, 3, padding=1),
            nn.BatchNorm2d(cnn_ch),
            nn.ReLU(),
        )
        self.spatial_res = nn.Sequential(*[ResBlock(cnn_ch) for _ in range(n_res)])
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 2))  # → (cnn_ch, 4, 2)
        spatial_flat = cnn_ch * 4 * 2  # 512

        # ── Hand encoder ─────────────────────────────────────────
        hand_in = 5 * embed_dim
        self.hand_mlp = nn.Sequential(
            nn.Linear(hand_in, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
        )

        # ── Global encoder ───────────────────────────────────────
        self.global_mlp = nn.Sequential(
            nn.Linear(10, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
        )

        # ── LSTM ─────────────────────────────────────────────────
        lstm_in = spatial_flat + mlp_hidden * 2
        self.lstm = nn.LSTMCell(lstm_in, lstm_hidden)

        # ── Card selector head ───────────────────────────────────
        self.card_head = nn.Linear(lstm_hidden, 5)

        # ── Per-card position heatmap head ───────────────────────
        # Produces (4, 32, 18) — one heatmap per hand slot
        self.pos_head = nn.Sequential(
            nn.Linear(lstm_hidden, cnn_ch * 4 * 2),
            nn.ReLU(),
        )
        self.pos_deconv = nn.Sequential(
            nn.ConvTranspose2d(cnn_ch, cnn_ch // 2, 4, stride=2, padding=1),  # → (8, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(cnn_ch // 2, cnn_ch // 4, 4, stride=2, padding=1),  # → (16, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(cnn_ch // 4, 4, (4, 5), stride=2, padding=(1, 2)),  # → (32, 18) approximately
        )
        # Final projection to exact size
        self.pos_final = nn.Conv2d(4, 4, 3, padding=1)

        # ── Value head ───────────────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def init_hidden(self, batch_size=1, device=None):
        """Return initial LSTM state (h, c)"""
        device = device or next(self.parameters()).device
        return (torch.zeros(batch_size, self.lstm_hidden, device=device),
                torch.zeros(batch_size, self.lstm_hidden, device=device))

    def forward(self, obs, hx):
        """
        Args:
            obs: dict with:
                arena:  (B, 16, 32, 18) float
                global: (B, 10) float
                hand:   (B, 5) long
            hx: (h, c) each (B, lstm_hidden)

        Returns:
            card_logits:  (B, 5)
            pos_heatmaps: (B, 4, 32, 18) — log-probs over arena for each hand slot
            value:        (B,)
            hx_new:       (h, c)
        """
        arena  = obs["arena"]    # (B, 16, 32, 18)
        glob   = obs["global"]   # (B, 10)
        hand   = obs["hand"]     # (B, 5)
        B = arena.shape[0]

        # ── Inject card embeddings into arena ────────────────────
        card_id_ch = arena[:, 2, :, :].long()          # (B, 32, 18)
        card_emb = self.card_embed(card_id_ch)          # (B, 32, 18, embed_dim)
        card_emb = card_emb.permute(0, 3, 1, 2)        # (B, embed_dim, 32, 18)
        arena_no_id = torch.cat([arena[:, :2], arena[:, 3:]], dim=1)  # (B, 15, 32, 18)
        spatial_in = torch.cat([arena_no_id, card_emb], dim=1)  # (B, 15+E, 32, 18)

        # ── CNN ──────────────────────────────────────────────────
        x = self.spatial_stem(spatial_in)
        x = self.spatial_res(x)                         # (B, cnn_ch, 32, 18)
        x_pool = self.spatial_pool(x).flatten(1)        # (B, spatial_flat)

        # ── Hand ─────────────────────────────────────────────────
        hand_emb = self.card_embed(hand).flatten(1)     # (B, 5*embed_dim)
        hand_feat = self.hand_mlp(hand_emb)             # (B, mlp_hidden)

        # ── Global ───────────────────────────────────────────────
        glob_feat = self.global_mlp(glob)               # (B, mlp_hidden)

        # ── LSTM ─────────────────────────────────────────────────
        lstm_in = torch.cat([x_pool, hand_feat, glob_feat], dim=1)
        hx_new = self.lstm(lstm_in, hx)
        h = hx_new[0]                                   # (B, lstm_hidden)

        # ── Heads ────────────────────────────────────────────────
        card_logits = self.card_head(h)                  # (B, 5)

        pos_feat = self.pos_head(h)                      # (B, cnn_ch*4*2)
        pos_feat = pos_feat.view(B, -1, 4, 2)           # (B, cnn_ch, 4, 2)
        pos_maps = self.pos_deconv(pos_feat)             # (B, 4, ~32, ~18)
        # Ensure exact output size
        pos_maps = F.interpolate(pos_maps, size=(ARENA_H, ARENA_W),
                                 mode='bilinear', align_corners=False)
        pos_maps = self.pos_final(pos_maps)              # (B, 4, 32, 18)

        value = self.value_head(h).squeeze(-1)           # (B,)

        return card_logits, pos_maps, value, hx_new

    def get_action(self, obs, hx, deterministic=False):
        """
        Sample an action from the policy.

        Returns: action_dict, log_prob, value, hx_new
        """
        card_logits, pos_maps, value, hx_new = self.forward(obs, hx)

        # Sample card
        card_dist = torch.distributions.Categorical(logits=card_logits)
        if deterministic:
            card_idx = card_logits.argmax(dim=-1)
        else:
            card_idx = card_dist.sample()

        card_log_prob = card_dist.log_prob(card_idx)     # (B,)

        # Sample position from the chosen card's heatmap
        B = card_idx.shape[0]
        safe_idx = card_idx.clamp(0, 3)
        chosen_map = pos_maps[torch.arange(B), safe_idx]  # (B, 32, 18)
        flat = chosen_map.flatten(1)                       # (B, 32*18)
        pos_dist = torch.distributions.Categorical(logits=flat)

        if deterministic:
            flat_idx = flat.argmax(dim=-1)
        else:
            flat_idx = pos_dist.sample()

        pos_log_prob = pos_dist.log_prob(flat_idx)

        # Convert flat index → (y, x) → tile centre
        ty = flat_idx // ARENA_W
        tx = flat_idx %  ARENA_W
        positions = torch.stack([tx.float() + 0.5, ty.float() + 0.5], dim=-1)

        # Combined log prob
        log_prob = card_log_prob + pos_log_prob

        action = {
            "card":     card_idx.cpu().numpy(),
            "position": positions.cpu().numpy(),
        }

        return action, log_prob, value, hx_new

    def evaluate_action(self, obs, hx, card_idx, flat_pos_idx):
        """
        Evaluate log_prob and entropy for given actions (for PPO update).
        """
        card_logits, pos_maps, value, hx_new = self.forward(obs, hx)

        card_dist = torch.distributions.Categorical(logits=card_logits)
        card_lp   = card_dist.log_prob(card_idx)
        card_ent  = card_dist.entropy()

        B = card_idx.shape[0]
        chosen_map = pos_maps[torch.arange(B), card_idx]
        flat = chosen_map.flatten(1)
        pos_dist = torch.distributions.Categorical(logits=flat)
        pos_lp   = pos_dist.log_prob(flat_pos_idx)
        pos_ent  = pos_dist.entropy()

        return card_lp + pos_lp, card_ent + pos_ent, value, hx_new


# ── Quick test ───────────────────────────────────────────────────
if __name__ == "__main__":
    model = ClashAgent()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    # Dummy obs
    obs = {
        "arena":  torch.zeros(1, NUM_TILE_FEATURES, ARENA_H, ARENA_W),
        "global": torch.randn(1, 10),
        "hand":   torch.randint(0, NUM_CARD_IDS, (1, 5)),
    }
    hx = model.init_hidden(1)
    action, lp, val, hx = model.get_action(obs, hx)
    print(f"Card: {action['card']}  Pos: {action['position']}  Value: {val.item():.3f}")
    print("✅ Model forward pass works!")
