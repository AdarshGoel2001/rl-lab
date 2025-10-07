🧠 World Models Component Analysis & Implementation Roadmap

  📊 Current Architecture Assessment

  ✅ What You Already Have

  Your codebase has an excellent modular foundation with:

  1. BaseParadigm - Clean abstraction for all agent types
  2. BaseWorldModelParadigm - Basic structure exists (src/paradigms/world_model/paradigm.py:24)
  3. Component Registry System - Plug-and-play architecture ready
  4. Factory Pattern - ComponentFactory for creating paradigms from configs

  Existing Components:
  - ✅ Encoders: MLP, Impala CNN
  - ✅ Dynamics: Deterministic MLP (basic)
  - ✅ Policy Heads: Categorical, Gaussian
  - ✅ Value Functions: MLP Critic
  - ✅ Representation Learners: Identity (passthrough)
  - ⚠️ Planners: Base interface only, no implementations

  ---
  🎯 What World Models Need

  Based on research of DreamerV3 (Nature 2025), MuZero, and TD-MPC, here are the required components:

  1. Core World Model Components (Missing)

  A. RSSM (Recurrent State Space Model) ⭐⭐⭐⭐⭐

  The heart of Dreamer - combines deterministic and stochastic latent states.

  Structure:
  # Components needed:
  - Representation model: h_t, z_t = encode(o_t, h_{t-1}, a_{t-1})
  - Transition predictor: z_t' = transition(h_t)  # prior
  - Posterior encoder: z_t = posterior(h_t, o_t)  # actual
  - Dynamics RNN: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})

  Current Gap: Only have deterministic dynamics, no RSSM implementation

  B. Reward Predictor ⭐⭐⭐⭐⭐

  Predicts rewards from latent states

  Missing Component:
  class RewardPredictor(nn.Module):
      """Predicts r_t from (h_t, z_t)"""
      def forward(self, state: torch.Tensor) -> Distribution
      def reward_loss(self, states, true_rewards) -> Dict[str, Tensor]

  Current Status: ❌ Not implemented

  C. Continue/Discount Predictor ⭐⭐⭐⭐

  Predicts episode termination

  Missing Component:
  class ContinuePredictor(nn.Module):
      """Predicts whether episode continues"""
      def forward(self, state: torch.Tensor) -> Distribution
      def continue_loss(self, states, dones) -> Dict[str, Tensor]

  Current Status: ❌ Not implemented

  ---
  2. Dynamics Models (Partially Complete)

  What You Have:

  - ✅ BaseDynamicsModel interface (src/components/dynamics/base.py:15)
  - ✅ DeterministicMLPDynamics (src/components/dynamics/deterministic_mlp.py:18)

  What's Missing:

  A. Stochastic Dynamics ⭐⭐⭐⭐⭐
  @register_dynamics_model("stochastic_mlp")
  class StochasticMLPDynamics(BaseDynamicsModel):
      """Returns distribution over next states, not deterministic"""
      # Uses Gaussian or Categorical distributions
      # Essential for RSSM

  B. RSSM Dynamics ⭐⭐⭐⭐⭐
  @register_dynamics_model("rssm")
  class RSSMDynamics(BaseDynamicsModel):
      """
      Recurrent State Space Model
      Combines deterministic recurrent state (h) 
      with stochastic latent state (z)
      """
      - GRU/LSTM for deterministic component
      - Stochastic latent variable model
      - Prior and posterior networks

  C. Ensemble Dynamics ⭐⭐⭐
  @register_dynamics_model("ensemble")
  class EnsembleDynamics:
      """Multiple dynamics models for uncertainty"""

  ---
  3. Representation Learners (Mostly Missing)

  What You Have:

  - ✅ BaseRepresentationLearner interface
  - ✅ IdentityRepresentationLearner (passthrough)

  What's Missing:

  A. VAE-style Representation ⭐⭐⭐⭐⭐
  @register_representation_learner("vae")
  class VAERepresentation(BaseRepresentationLearner):
      """Variational autoencoder for latent representations"""
      def encode(self, features) -> (mu, logvar)
      def decode(self, z) -> reconstruction
      def representation_loss(self, features) -> (recon_loss + kl_loss)

  B. Contrastive Representation ⭐⭐⭐⭐
  @register_representation_learner("contrastive")
  class ContrastiveRepresentation(BaseRepresentationLearner):
      """Contrastive learning (MoCo, SimCLR style)"""

  C. Dreamer-style RSSM Representation ⭐⭐⭐⭐⭐
  @register_representation_learner("rssm_vae")
  class RSSMRepresentation(BaseRepresentationLearner):
      """
      Learns h (deterministic) and z (stochastic) jointly
      Integrates with RSSM dynamics
      """

  ---
  4. Planners (Empty - Need Everything)

  Current Status:

  - ✅ BasePlanner interface (src/components/planners/base.py:19)
  - ❌ No implementations at all

  What's Needed:

  A. MCTS Planner ⭐⭐⭐⭐⭐
  @register_planner("mcts")
  class MCTSPlanner(BasePlanner):
      """Monte Carlo Tree Search (for MuZero)"""
      def plan(self, state, dynamics, value) -> action_dist
      # Requires:
      - Tree search with dynamics model
      - UCB action selection
      - Value/policy backup

  B. Shooting Methods ⭐⭐⭐⭐
  @register_planner("cem")
  class CEMPlanner(BasePlanner):
      """Cross-Entropy Method sampling-based planning"""

  @register_planner("mppi")
  class MPPIPlanner(BasePlanner):
      """Model Predictive Path Integral"""

  C. Gradient-Based Planning ⭐⭐⭐⭐
  @register_planner("gradient")
  class GradientPlanner(BasePlanner):
      """Backpropagate through dynamics (TD-MPC style)"""

  ---
  5. Encoders (Need Visual Encoders)

  What You Have:

  - ✅ Simple MLP encoder
  - ✅ Impala CNN (for Atari)

  What's Missing for World Models:

  A. ResNet Encoder ⭐⭐⭐⭐
  @register_encoder("resnet")
  class ResNetEncoder(BaseEncoder):
      """ResNet blocks for visual observations"""

  B. Transformer Encoder ⭐⭐⭐
  @register_encoder("transformer")
  class TransformerEncoder(BaseEncoder):
      """For TransDreamer variant"""

  ---
  🔍 Gap Analysis Summary

  | Component Type      | Have              | Need                        | Priority |
  |---------------------|-------------------|-----------------------------|----------|
  | Dynamics Models     | Deterministic MLP | RSSM, Stochastic, Ensemble  | ⭐⭐⭐⭐⭐    |
  | Reward Predictors   | None              | Reward MLP/Ensemble         | ⭐⭐⭐⭐⭐    |
  | Continue Predictors | None              | Continue/Discount predictor | ⭐⭐⭐⭐⭐    |
  | Representation      | Identity only     | VAE, RSSM-VAE, Contrastive  | ⭐⭐⭐⭐⭐    |
  | Planners            | Base interface    | MCTS, CEM, MPPI, Gradient   | ⭐⭐⭐⭐     |
  | Encoders            | MLP, CNN          | ResNet, Transformer         | ⭐⭐⭐      |

  ---
  🗺️ Implementation Roadmap

  Phase 1: Essential Predictors (1 week)

  Create new component types in your architecture:

  1.1 Reward Predictor Component

  src/components/reward_predictors/
  ├── __init__.py
  ├── base.py              # BaseRewardPredictor
  └── mlp_reward.py        # MLPRewardPredictor

  Base Interface:
  # src/components/reward_predictors/base.py
  class BaseRewardPredictor(nn.Module, ABC):
      @abstractmethod
      def forward(self, state: Tensor) -> Distribution

      def reward_loss(self, states: Tensor, rewards: Tensor) -> Dict:
          """MSE or NLL loss"""

  Implementation:
  # src/components/reward_predictors/mlp_reward.py
  @register_reward_predictor("mlp")
  class MLPRewardPredictor(BaseRewardPredictor):
      def _build_model(self):
          self.network = MLP(state_dim, hidden_dims, 1)

      def forward(self, state):
          mean = self.network(state)
          return Normal(mean, std)

  1.2 Continue Predictor Component

  src/components/continue_predictors/
  ├── __init__.py
  ├── base.py              # BaseContinuePredictor
  └── mlp_continue.py      # MLPContinuePredictor

  Interface:
  class BaseContinuePredictor(nn.Module, ABC):
      @abstractmethod
      def forward(self, state: Tensor) -> Distribution

      def continue_loss(self, states: Tensor, continues: Tensor) -> Dict:
          """Binary cross-entropy loss"""

  1.3 Update Registry

  # src/utils/registry.py
  REWARD_PREDICTOR_REGISTRY = {}
  CONTINUE_PREDICTOR_REGISTRY = {}

  def register_reward_predictor(name: str) -> Callable: ...
  def register_continue_predictor(name: str) -> Callable: ...

  ---
  Phase 2: Stochastic Dynamics (1 week)

  2.1 Stochastic MLP Dynamics

  # src/components/dynamics/stochastic_mlp.py
  @register_dynamics_model("stochastic_mlp")
  class StochasticMLPDynamics(BaseDynamicsModel):
      """Gaussian dynamics model"""

      def _build_model(self):
          self.mean_net = MLP(...)
          self.std_net = MLP(...) or nn.Parameter(...)

      def forward(self, state, action):
          mean = self.mean_net(torch.cat([state, action], -1))
          std = F.softplus(self.std_net(...)) + min_std
          return Normal(mean, std)

  2.2 Categorical Dynamics

  @register_dynamics_model("categorical")
  class CategoricalDynamics(BaseDynamicsModel):
      """Discrete latent dynamics (for Dreamer)"""

      def forward(self, state, action):
          logits = self.network(torch.cat([state, action], -1))
          return OneHotCategorical(logits=logits)

  ---
  Phase 3: RSSM Implementation (2 weeks) ⭐⭐⭐⭐⭐

  This is the core of DreamerV3.

  3.1 RSSM Representation Learner

  # src/components/representation_learners/rssm.py
  @register_representation_learner("rssm")
  class RSSMRepresentation(BaseRepresentationLearner):
      """
      Recurrent State Space Model representation
      Combines deterministic (h) and stochastic (z) states
      """

      def _build_learner(self):
          # Deterministic path
          self.rnn = nn.GRUCell(stoch_dim + action_dim, deter_dim)

          # Stochastic path
          self.prior_net = MLP(deter_dim, stoch_dim * 32)  # 32 classes
          self.posterior_net = MLP(deter_dim + embed_dim, stoch_dim * 32)

          # Decoder for reconstruction
          self.decoder = nn.Sequential(...)

      def encode(self, features, prev_action, prev_state):
          """
          Returns: (h_t, z_t) where
          - h_t: deterministic recurrent state
          - z_t: stochastic latent state
          """
          h_prev, z_prev = prev_state

          # Deterministic update
          h = self.rnn(torch.cat([z_prev, prev_action], -1), h_prev)

          # Stochastic posterior (uses observation)
          z_posterior = self.posterior_net(torch.cat([h, features], -1))
          z = OneHotCategorical(logits=z_posterior).sample()

          return torch.cat([h, z], -1)

      def imagine(self, prev_state, action):
          """Imagine next state without observation (uses prior)"""
          h_prev, z_prev = prev_state

          h = self.rnn(torch.cat([z_prev, action], -1), h_prev)
          z_prior = self.prior_net(h)
          z = OneHotCategorical(logits=z_prior).sample()

          return torch.cat([h, z], -1)

      def representation_loss(self, features, actions, prev_states):
          """KL divergence between prior and posterior"""
          # ... KL loss + reconstruction loss

  3.2 RSSM Dynamics Model

  # src/components/dynamics/rssm_dynamics.py
  @register_dynamics_model("rssm")
  class RSSMDynamics(BaseDynamicsModel):
      """
      Uses RSSM representation for dynamics
      Wraps the RSSMRepresentation's imagine() method
      """

      def __init__(self, config, rssm_repr: RSSMRepresentation):
          self.rssm = rssm_repr

      def forward(self, state, action):
          """Predict next state using RSSM prior"""
          return self.rssm.imagine(state, action)

  ---
  Phase 4: Complete DreamerV3 Paradigm (1-2 weeks)

  4.1 Enhanced WorldModelParadigm

  # src/paradigms/dreamer.py
  @register_paradigm("dreamer")
  class DreamerParadigm(BaseParadigm):
      """
      DreamerV3 implementation
      """

      def __init__(self, 
                   encoder: BaseEncoder,
                   rssm: RSSMRepresentation,
                   reward_predictor: BaseRewardPredictor,
                   continue_predictor: BaseContinuePredictor,
                   actor: BasePolicyHead,
                   critic: BaseValueFunction,
                   config: Dict):

          self.encoder = encoder
          self.rssm = rssm
          self.reward_pred = reward_predictor
          self.continue_pred = continue_predictor
          self.actor = actor
          self.critic = critic

          # Separate optimizers
          self.world_model_opt = Adam(
              list(encoder.parameters()) +
              list(rssm.parameters()) +
              list(reward_pred.parameters()) +
              list(continue_pred.parameters())
          )
          self.actor_opt = Adam(actor.parameters())
          self.critic_opt = Adam(critic.parameters())

      def compute_loss(self, batch):
          """
          DreamerV3 loss computation
          """
          obs = batch['observations']
          actions = batch['actions']
          rewards = batch['rewards']
          continues = 1 - batch['dones'].float()

          # === World Model Training ===
          # 1. Encode observations
          embeds = self.encoder(obs)

          # 2. Compute RSSM states
          states = self.rssm.encode_sequence(embeds, actions)

          # 3. World model losses
          wm_losses = {}

          # Representation loss (KL + reconstruction)
          wm_losses.update(self.rssm.representation_loss(embeds, actions))

          # Reward prediction loss
          pred_rewards = self.reward_pred(states)
          wm_losses['reward_loss'] = -pred_rewards.log_prob(rewards).mean()

          # Continue prediction loss
          pred_continues = self.continue_pred(states)
          wm_losses['continue_loss'] = F.binary_cross_entropy_with_logits(
              pred_continues.logits, continues
          )

          # === Behavior Learning (Actor-Critic) ===
          # Imagination rollout
          imag_states, imag_actions, imag_rewards, imag_continues = \
              self.imagine_trajectory(states.detach(), horizon=15)

          # Compute returns using lambda-returns
          returns = self.compute_lambda_returns(
              imag_rewards,
              self.critic(imag_states),
              imag_continues
          )

          # Actor loss (maximize returns)
          actor_loss = -returns.mean()
          wm_losses['actor_loss'] = actor_loss

          # Critic loss (MSE)
          critic_loss = F.mse_loss(self.critic(imag_states), returns.detach())
          wm_losses['critic_loss'] = critic_loss

          return wm_losses

      def imagine_trajectory(self, initial_states, horizon):
          """Rollout in imagination using world model"""
          states = [initial_states]
          actions = []
          rewards = []
          continues = []

          state = initial_states
          for t in range(horizon):
              # Sample action from policy
              action_dist = self.actor(state.detach())
              action = action_dist.sample()

              # Predict next state
              next_state = self.rssm.imagine(state, action)

              # Predict reward and continue
              reward = self.reward_pred(state).mean
              continue_prob = torch.sigmoid(self.continue_pred(state).logits)

              states.append(next_state)
              actions.append(action)
              rewards.append(reward)
              continues.append(continue_prob)

              state = next_state

          return (
              torch.stack(states[:-1], 1),
              torch.stack(actions, 1),
              torch.stack(rewards, 1),
              torch.stack(continues, 1)
          )

  ---
  Phase 5: MuZero Components (2 weeks)

  5.1 MuZero Representation

  @register_representation_learner("muzero")
  class MuZeroRepresentation(BaseRepresentationLearner):
      """
      MuZero h-function: observation -> hidden state
      """
      def encode(self, observation):
          return self.network(observation)

  5.2 MuZero Dynamics

  @register_dynamics_model("muzero")
  class MuZeroDynamics(BaseDynamicsModel):
      """
      MuZero g-function: (hidden_state, action) -> (next_hidden_state, reward)
      """
      def forward(self, state, action):
          next_state = self.state_network(torch.cat([state, action], -1))
          reward = self.reward_network(torch.cat([state, action], -1))
          return next_state, reward

  5.3 MCTS Planner

  @register_planner("mcts")
  class MCTSPlanner(BasePlanner):
      """
      Monte Carlo Tree Search for MuZero
      """
      def plan(self, state, dynamics, value_fn, policy_fn):
          # Tree search with UCB
          for simulation in range(num_simulations):
              node = self.select(root, ucb_c)
              value = self.expand_and_evaluate(node)
              self.backup(node, value)

          # Return improved policy
          return self.get_action_distribution(root)

  ---
  Phase 6: Advanced Components (Optional, 1-2 weeks each)

  6.1 Ensemble Methods

  @register_dynamics_model("ensemble")
  @register_reward_predictor("ensemble")

  6.2 Transformer-based Components

  @register_encoder("transformer")
  @register_dynamics_model("transformer_rssm")

  6.3 Additional Planners

  @register_planner("cem")  # Cross-Entropy Method
  @register_planner("mppi")  # Model Predictive Path Integral
  @register_planner("td_mpc")  # TD-MPC style gradient planning

  ---
  📦 Required Registry Updates

  Update src/utils/registry.py:

  # Add new registries
  REWARD_PREDICTOR_REGISTRY: Dict[str, Type] = {}
  CONTINUE_PREDICTOR_REGISTRY: Dict[str, Type] = {}

  # Add registration functions
  def register_reward_predictor(name: str) -> Callable: ...
  def register_continue_predictor(name: str) -> Callable: ...

  # Add getter functions
  def get_reward_predictor(name: str) -> Type: ...
  def get_continue_predictor(name: str) -> Type: ...

  # Update list_registered_components()
  def list_registered_components() -> Dict[str, list]:
      return {
          ...
          'reward_predictors': list(REWARD_PREDICTOR_REGISTRY.keys()),
          'continue_predictors': list(CONTINUE_PREDICTOR_REGISTRY.keys()),
      }

  # Update auto_import_modules()
  module_dirs = [
      ...
      'components.reward_predictors',
      'components.continue_predictors',
  ]

  ---
  📝 Example Config for Dreamer

  experiment:
    name: dreamer_cartpole
    paradigm: dreamer

  components:
    encoder:
      type: mlp
      config:
        hidden_dims: [256, 256]
        activation: elu

    representation_learner:
      type: rssm
      config:
        deter_dim: 256        # Deterministic state size
        stoch_dim: 32         # Stochastic state size (32 classes)
        hidden_dim: 256

    reward_predictor:
      type: mlp
      config:
        hidden_dims: [256, 256]
        distribution: normal

    continue_predictor:
      type: mlp
      config:
        hidden_dims: [256, 256]

    policy_head:
      type: gaussian_mlp
      config:
        hidden_dims: [256, 256]

    value_function:
      type: mlp_critic
      config:
        hidden_dims: [256, 256]

  algorithm:
    name: dreamer
    world_model_lr: 1e-4
    actor_lr: 3e-5
    critic_lr: 3e-5
    imagination_horizon: 15
    lambda_return: 0.95

  ---
  ⚡ Quick Start Priorities

  If you want to get started quickly:

  Week 1: Minimum Viable World Model

  1. Create reward predictor component (1 day)
  2. Create continue predictor component (1 day)
  3. Create stochastic MLP dynamics (1 day)
  4. Update world model paradigm to use all three (1 day)
  5. Test with simple environment (1 day)

  Week 2-3: RSSM + DreamerV3

  1. Implement RSSM representation learner (3 days)
  2. Integrate RSSM with dynamics (2 days)
  3. Implement DreamerParadigm (3 days)
  4. Training + debugging (2 days)

  Week 4: MuZero (if needed)

  1. MuZero representation (1 day)
  2. MuZero dynamics (1 day)
  3. MCTS planner (3 days)
  4. Integration + testing (2 days)

  ---
  🎯 Critical Success Factors

  1. Reward & Continue Predictors - Absolutely essential, start here
  2. RSSM - The core innovation of Dreamer, most complex component
  3. Imagination Training Loop - Gradient flow through imagined trajectories
  4. Separate Optimizers - World model, actor, critic need separate optimizers
  5. Proper Loss Balancing - DreamerV3 uses specific loss scales/normalization

  ---

● 💡 Summary: What You Need to Add

  Your infrastructure is 90% ready for world models! You just need to fill in these component types:

  New Component Types Needed:

  1. ✅ Registry updates
  2. 🆕 Reward Predictors - New directory: src/components/reward_predictors/
  3. 🆕 Continue Predictors - New directory: src/components/continue_predictors/
  4. ⬆️ Stochastic Dynamics - Add to existing src/components/dynamics/
  5. ⬆️ RSSM Representation - Add to existing src/components/representation_learners/
  6. ⬆️ Planner Implementations - Add to existing src/components/planners/

  Your Modular Architecture's Strengths:

  - ✅ Registry system perfect for adding components
  - ✅ Factory pattern ready for world model configs
  - ✅ BaseParadigm extensible enough
  - ✅ Component interfaces well-designed

  The Path Forward:

  Option A: Quick Dreamer (3 weeks)
  - Add reward/continue predictors
  - Implement RSSM
  - Create DreamerParadigm
  - Start training!

  Option B: Full Suite (6-8 weeks)
  - All of Option A
  - Add MuZero components
  - Implement MCTS
  - Add all planner variants
  - Transformer encoders

  Your codebase is exceptionally well-structured for this - the modular component system is exactly what world models need. You just need to implement the missing
  pieces!
