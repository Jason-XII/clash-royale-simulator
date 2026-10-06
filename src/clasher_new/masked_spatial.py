"""The PPO policy: pick WAIT, a card, or a bank target; for a card, then a tile.

Observations must include `legal_mask` (4, 32, 18): which tiles each hand slot can
use right now. Bank targets already reached are masked using `elixir`.
Sampling, log-probabilities and entropy all use the same masks.
Checkpoints from before banking (5 action slots) still load and never bank.

Policies built with the memory inputs are recurrent: an LSTM over decisions sits
between the encoder and the heads. Its state is carried three ways:
- collecting: the running state lives in `state_in`, the rollout callback stores
  it per step and advances it (parallel_rollout.Events);
- training: minibatches are chunks of consecutive decisions, with each chunk's
  starting state and the episode starts added to the observations;
- playing: `predict(obs, state)` takes and returns the state, as in sb3-contrib.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment import BANK_TARGETS, CARD_SLOTS, OPPONENT_PLAYS, entity_names

EMBED = 8          # entity embedding size inside the board encoder
MEMORY_FRAMES = 2  # board frames the encoder reads when the memory inputs exist
CARD_EMBED = 16    # card embedding size in the policy heads
TILES = 32 * 18
LSTM_HIDDEN = 256  # = the encoder's features_dim, which the heads expect
RECURRENT_KEYS = ("lstm_state", "episode_start")  # training-only observation extras


class BoardEncoder(BaseFeaturesExtractor):
    """CNN over the stacked board frames plus hand, elixir and clock.

    With the memory inputs (`queue`, `opponent_elixir`, `decision_gap`) in the
    observation space, it reads only the last MEMORY_FRAMES frames and adds those
    inputs. With `opponent_plays` it also reads the opponent's recent plays.
    Checkpoints from before either keep their layout.

    Also keeps the first full-resolution conv activation in `self.spatial`
    (B, 32, 32, 18) for the tile scorer.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.entity_embedding = nn.Embedding(len(entity_names), EMBED)
        self.memory = "queue" in observation_space.spaces
        self.frames = MEMORY_FRAMES if self.memory else observation_space["grid"].shape[0]
        # 15 raw channels: id -> embedding, type -> one-hot(4), 13 numeric stay.
        in_channels = self.frames * (13 + EMBED + 4)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, in_channels, 32, 18)).shape[1]
        # hand embeddings + elixir + one-hot phase + time left
        # (+ queue embeddings, opponent elixir, decision gap)
        memory_inputs = 3 * EMBED + 2 if self.memory else 0
        self.history = "opponent_plays" in observation_space.spaces
        history_inputs = OPPONENT_PLAYS * (EMBED + 3) if self.history else 0
        self.fc = nn.Linear(cnn_out + 5 * EMBED + 1 + 4 + 1 + memory_inputs + history_inputs, features_dim)

    def forward(self, obs):
        grid = obs["grid"][:, -self.frames:]                  # (B, F, 32, 18, 15)
        ids = self.entity_embedding(grid[..., 0].long())
        card_type = F.one_hot(grid[..., 1].long(), num_classes=4).float()
        x = torch.cat([grid[..., 2:], ids, card_type], dim=-1)
        x = x.permute(0, 1, 4, 2, 3).flatten(1, 2).float()   # (B, F*C, 32, 18)
        self.spatial = self.cnn[1](self.cnn[0](x))
        board = self.cnn[2:](self.spatial)
        hand = self.entity_embedding(obs["hand"].long()).flatten(1)
        inputs = [board, hand, obs["elixir"].float(), obs["phase"].float().flatten(1),
                  obs["time_till_next_phase"].float()]
        if self.memory:
            inputs += [self.entity_embedding(obs["queue"].long()).flatten(1),
                       obs["opponent_elixir"].float(),
                       obs["decision_gap"].float() / 10]   # ponytail: banking gaps reach ~30 s
        if self.history:
            plays = obs["opponent_plays"].float()                     # (B, 8, card/age/x/y)
            inputs += [self.entity_embedding(plays[..., 0].long()).flatten(1),
                       (plays[..., 1:] / plays.new_tensor([30, 18, 32])).flatten(1)]
        return torch.relu(self.fc(torch.cat(inputs, dim=1)))


class MaskedSpatialPolicy(MultiInputActorCriticPolicy):
    WAIT_PLAY_ENTROPY_WEIGHT = 0.10
    PLACEMENT_ENTROPY_WEIGHT = 0.25
    BANK_ENTROPY_WEIGHT = 1.00

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        kwargs.pop("allow_saving", None)  # recorded by older checkpoints; feature removed
        kwargs.setdefault("features_extractor_class", BoardEncoder)
        if not kwargs.get("share_features_extractor", True):
            raise ValueError("MaskedSpatialPolicy requires a shared feature extractor")
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        latent = self.global_dim = self.mlp_extractor.latent_dim_pi
        self.n_banks = int(action_space.nvec[0]) - 1 - CARD_SLOTS
        # Replaces SB3's MultiDiscrete head. Keep this registration order: the
        # saved optimizer state is matched to parameters by position.
        self.action_net = nn.Identity()
        self.card_embedding = nn.Embedding(len(entity_names), CARD_EMBED)
        self.card_scorer = nn.Sequential(nn.Linear(latent + CARD_EMBED, 64), nn.Tanh(), nn.Linear(64, 1))
        self.noop_head = nn.Linear(latent, 1)
        self.placement_context = nn.Linear(latent + CARD_EMBED, 32)
        self.tile_scorer = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )
        if self.n_banks:
            self.bank_head = nn.Linear(latent, self.n_banks)
            self.register_buffer("bank_targets", torch.tensor(BANK_TARGETS[:self.n_banks],
                                                              dtype=torch.float32), persistent=False)
        self.recurrent = self.features_extractor.memory
        if self.recurrent:
            self.lstm = nn.LSTMCell(self.features_dim, LSTM_HIDDEN)
        self.state_in = self.next_state = None   # (B, 2, LSTM_HIDDEN): h and c
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Older checkpoints carry an unused elixir-cost buffer; legal_mask replaced it.
        state_dict = {k: v for k, v in state_dict.items() if k != "card_cost_table"}
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Returns (actions, state). Pass the returned state back on the next
        decision of the same game; None (or episode_start) starts a new game."""
        if not self.recurrent:
            return super().predict(observation, state, episode_start, deterministic)
        hand = np.asarray(observation["hand"])
        rows = len(hand) if hand.ndim == 2 else 1
        if state is None:
            state = np.zeros((rows, 2, LSTM_HIDDEN), dtype=np.float32)
        if episode_start is not None:
            state = state * (1 - np.asarray(episode_start, dtype=np.float32).reshape(rows, 1, 1))
        self.state_in = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        actions, _ = super().predict(observation, None, None, deterministic)
        return actions, self.next_state.cpu().numpy()

    def predict_values(self, obs):
        return self.value_net(self._latents(obs)[2])

    def obs_to_tensor(self, observation):
        # Checkpoints from before the memory inputs reject observation keys they don't know,
        # and newer ones are trained on only the last frames of the environment's grid.
        if isinstance(observation, dict):
            observation = {key: value for key, value in observation.items()
                           if key in self.observation_space.spaces}
            frames = self.observation_space["grid"].shape[0]
            observation["grid"] = np.asarray(observation["grid"])[..., -frames:, :, :, :]
        return super().obs_to_tensor(observation)

    # --- distributions -------------------------------------------------------

    def _latents(self, obs):
        """Returns (latent_pi, spatial map, latent_vf)."""
        features = self.extract_features({k: v for k, v in obs.items() if k not in RECURRENT_KEYS})
        if self.recurrent:
            features = self._recur(features, obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        return latent_pi, self.features_extractor.spatial, latent_vf

    def _recur(self, features, obs):
        """The LSTM over decisions. Training observations carry each chunk's starting
        state and per-step episode starts (features are chunk-major); otherwise this
        is one step from `state_in`. Leaves the final state in `next_state`."""
        if "lstm_state" in obs:
            state, starts = obs["lstm_state"], obs["episode_start"].float()
        else:
            if self.state_in is None or len(self.state_in) != len(features):
                self.state_in = features.new_zeros(len(features), 2, LSTM_HIDDEN)
            state, starts = self.state_in, features.new_zeros(len(features))
        chunks = len(state)
        x, starts = features.view(chunks, -1, features.shape[1]), starts.view(chunks, -1)
        h, c = state[:, 0], state[:, 1]
        outputs = []
        for step in range(x.shape[1]):
            keep = 1 - starts[:, step:step + 1]
            h, c = self.lstm(x[:, step], (h * keep, c * keep))
            outputs.append(h)
        self.next_state = torch.stack((h, c), dim=1).detach()
        return torch.stack(outputs, dim=1).reshape(len(features), -1)

    def _choice_distribution(self, latent_pi, obs):
        """Categorical over [WAIT, card slot 1..4, bank target 1..n]."""
        hand = obs["hand"][:, :CARD_SLOTS].long()
        state = latent_pi.unsqueeze(1).expand(-1, CARD_SLOTS, -1)
        cards = self.card_scorer(torch.cat((state, self.card_embedding(hand)), dim=2)).squeeze(2)
        logits = [self.noop_head(latent_pi), cards.masked_fill(~self._legal_slots(obs), -1e9)]
        if self.n_banks:
            reached = obs["elixir"].float() >= self.bank_targets
            logits.append(self.bank_head(latent_pi).masked_fill(reached, -1e9))
        return Categorical(logits=torch.cat(logits, dim=1))

    def _placement_distribution(self, latent_pi, spatial, obs, slot):
        """Categorical over the 576 tiles for hand slot `slot` (0-based, (B,)).

        A slot with no legal tile (WAIT, banking, unaffordable) gets a single
        placeholder tile, so it contributes zero log-probability and entropy.
        """
        card = obs["hand"][:, :CARD_SLOTS].long().gather(1, slot[:, None]).squeeze(1)
        context = self.placement_context(torch.cat((latent_pi, self.card_embedding(card)), dim=1))
        logits = self.tile_scorer(spatial + context[:, :, None, None]).flatten(1)
        tiles = obs["legal_mask"].flatten(2).bool()[torch.arange(len(slot), device=slot.device), slot].clone()
        tiles[~tiles.any(dim=1), 0] = True
        return Categorical(logits=logits.masked_fill(~tiles, -1e9))

    @staticmethod
    def _legal_slots(obs):
        return obs["legal_mask"].flatten(2).bool().any(dim=2)

    @staticmethod
    def _plays(choice):
        return (choice >= 1) & (choice <= CARD_SLOTS)

    # --- SB3 interface -------------------------------------------------------

    def _sample_action(self, obs, deterministic=False):
        latent_pi, spatial, latent_vf = self._latents(obs)
        choices = self._choice_distribution(latent_pi, obs)
        choice = choices.logits.argmax(1) if deterministic else choices.sample()
        place = self._placement_distribution(latent_pi, spatial, obs, (choice - 1).clamp(0, CARD_SLOTS - 1))
        tile = place.logits.argmax(1) if deterministic else place.sample()
        play = self._plays(choice)
        tile = torch.where(play, tile, torch.zeros_like(tile))
        log_prob = choices.log_prob(choice) + play.float() * place.log_prob(tile)
        return torch.stack((choice, tile // 18, tile % 18), dim=1), log_prob, latent_vf

    def forward(self, obs, deterministic=False):
        actions, log_prob, latent_vf = self._sample_action(obs, deterministic)
        return actions, self.value_net(latent_vf), log_prob

    def _predict(self, observation, deterministic=False):
        return self._sample_action(observation, deterministic)[0]

    def evaluate_actions(self, obs, actions):
        latent_pi, spatial, latent_vf = self._latents(obs)
        actions = actions.long()
        choice, tile = actions[:, 0], actions[:, 1] * 18 + actions[:, 2]
        choices = self._choice_distribution(latent_pi, obs)
        place = self._placement_distribution(latent_pi, spatial, obs, (choice - 1).clamp(0, CARD_SLOTS - 1))
        log_prob = choices.log_prob(choice) + self._plays(choice).float() * place.log_prob(tile)
        entropy = self._conditional_entropy(latent_pi, spatial, choices, obs)
        return self.value_net(latent_vf), log_prob, entropy

    # --- exploration bonus ---------------------------------------------------

    def _conditional_entropy(self, latent_pi, spatial, choices, obs):
        """Separately normalized entropies of each decision in the hierarchy:
        hold (WAIT/bank) vs PLAY, which hold option, which card, which tile.

        Each lower choice is conditional on the branch above it, so its bonus never
        rewards playing or holding by itself. A term is zero where that choice is
        forced, and each term is rescaled so forced states do not dilute it.
        """
        def rescale(eligible):
            return eligible.float() * eligible.numel() / eligible.sum().clamp(min=1)

        logits = choices.logits
        card_logits = logits[:, 1:1 + CARD_SLOTS]
        hold_logits = torch.cat((logits[:, :1], logits[:, 1 + CARD_SLOTS:]), dim=1)
        legal = self._legal_slots(obs)
        count = legal.sum(dim=1)
        decision = count >= 1
        given_play = Categorical(logits=card_logits)
        given_hold = Categorical(logits=hold_logits)
        hold_play = Categorical(logits=torch.stack(
            (torch.logsumexp(hold_logits, dim=1), torch.logsumexp(card_logits, dim=1)), dim=1))

        wait_play_entropy = hold_play.entropy() / np.log(2) * rescale(decision)
        card_entropy = given_play.entropy() / np.log(CARD_SLOTS) * rescale(count >= 2)
        bank_entropy = torch.zeros_like(card_entropy)
        if self.n_banks:
            hold_options = 1 + (obs["elixir"].float() < self.bank_targets).sum(dim=1)
            bank_entropy = (given_hold.entropy() / np.log(1 + self.n_banks)
                            * rescale(decision & (hold_options >= 2)))
        placement = torch.zeros_like(card_entropy)
        for slot in range(CARD_SLOTS):
            slots = torch.full_like(count, slot)
            tile_entropy = self._placement_distribution(latent_pi, spatial, obs, slots).entropy()
            placement += given_play.probs[:, slot] * tile_entropy
        placement_entropy = placement / np.log(TILES) * rescale(decision)

        objective = (self.WAIT_PLAY_ENTROPY_WEIGHT * wait_play_entropy + card_entropy
                     + self.BANK_ENTROPY_WEIGHT * bank_entropy
                     + self.PLACEMENT_ENTROPY_WEIGHT * placement_entropy)
        play_probability = choices.probs[:, 1:1 + CARD_SLOTS].sum(dim=1)
        self.entropy_diagnostics = {
            "wait_play": wait_play_entropy.mean().detach(),
            "actionable_card": card_entropy.mean().detach(),
            "bank_choice": bank_entropy.mean().detach(),
            "legal_placement": placement_entropy.mean().detach(),
            "joint_action": (choices.entropy() + play_probability * placement).mean().detach(),
            "objective": objective.mean().detach(),
        }
        return objective
