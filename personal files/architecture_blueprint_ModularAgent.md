# Modular Agent Architecture Blueprint: The Future of Robotics Research Infrastructure

## Executive Summary

This document defines the architectural blueprint for a fully modular robotics research platform where any component can be swapped with any other compatible component. The goal is to create a "LEGO system" for AI research where encoders, dynamics models, planning systems, and policy heads can be freely combined to create novel agent architectures.

**Core Design Principle**: Every component is a plug-and-play module that implements standardized interfaces, enabling researchers to focus on innovation rather than integration.

---

## The Component Taxonomy: Building Blocks of Intelligence

### 1. Perception Components (How agents see the world)

**Role**: Transform raw sensory input into useful representations

#### 1.1 Encoders
```python
class BaseEncoder(nn.Module):
    """Transforms observations to feature representations"""
    def forward(self, observations: Dict[str, Tensor]) -> Tensor:
        """obs -> features"""
        pass

    @property
    def output_dim(self) -> int:
        """Dimensionality of output features"""
        pass

    @property
    def supports_sequences(self) -> bool:
        """Whether encoder handles temporal sequences"""
        pass
```

**Registry of Encoders**:
- `impala_cnn`: Standard CNN for pixel observations
- `resnet_backbone`: ImageNet-pretrained CNN
- `dino_v2`: Self-supervised vision transformer (frozen/finetunable)
- `clip_vision`: CLIP vision encoder for language-conditioned tasks
- `proprioception_mlp`: Simple MLP for robot joint states
- `multimodal_fusion`: Combines vision + proprioception + audio
- `temporal_cnn`: 3D CNN for video sequences
- `point_cloud_net`: Point cloud processing (PointNet-style)

**Configuration Examples**:
```yaml
encoder:
  type: "dino_v2"
  config:
    model_size: "base"  # base, large, giant
    frozen: true
    fine_tune_layers: 2  # Only fine-tune last 2 layers

encoder:
  type: "multimodal_fusion"
  config:
    vision: {type: "clip_vision", frozen: true}
    proprioception: {type: "proprioception_mlp", hidden_dims: [256, 128]}
    fusion_method: "concatenate"  # or "cross_attention"
```

#### 1.2 Representation Learners
```python
class BaseRepresentationLearner(nn.Module):
    """Learns structured representations beyond raw features"""
    def encode(self, features: Tensor) -> Tensor:
        """features -> structured representation"""
        pass

    def decode(self, representation: Tensor) -> Tensor:
        """representation -> reconstructed features"""
        pass

    def representation_loss(self, features: Tensor) -> Tensor:
        """Self-supervised learning objective"""
        pass
```

**Registry of Representation Learners**:
- `vae`: Variational autoencoder for stochastic representations
- `beta_vae`: Disentangled representations
- `world_model_encoder`: Joint encoder-dynamics learning (Dreamer-style)
- `contrastive_learner`: SimCLR/MoCo-style contrastive learning
- `masked_autoencoder`: MAE-style reconstruction
- `identity`: Pass-through (no representation learning)

### 2. Reasoning Components (How agents think about the world)

#### 2.1 Dynamics Models
```python
class BaseDynamicsModel(nn.Module):
    """Predicts how the world evolves"""
    def forward(self, state: Tensor, action: Tensor) -> Distribution:
        """(state, action) -> next_state_distribution"""
        pass

    def predict_sequence(self, initial_state: Tensor, actions: Tensor) -> Tensor:
        """Rollout sequence of predictions"""
        pass

    def dynamics_loss(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """Learning objective for dynamics"""
        pass
```

**Registry of Dynamics Models**:
- `deterministic_mlp`: Simple deterministic dynamics
- `stochastic_mlp`: Gaussian dynamics model
- `dreamer_rssm`: Recurrent State Space Model (Dreamer-style)
- `transformer_dynamics`: Attention-based dynamics
- `mamba_ssm`: State Space Model dynamics
- `ensemble_dynamics`: Uncertainty-aware ensemble
- `linear_dynamics`: Simple linear dynamics (for control theory)
- `physics_informed`: Incorporates known physics constraints

#### 2.2 Value Functions
```python
class BaseValueFunction(nn.Module):
    """Estimates state or state-action values"""
    def forward(self, state: Tensor, action: Optional[Tensor] = None) -> Tensor:
        """state [, action] -> value"""
        pass

    def value_loss(self, states: Tensor, targets: Tensor) -> Tensor:
        """Learning objective for value function"""
        pass
```

**Registry of Value Functions**:
- `mlp_critic`: Standard MLP value function
- `distributional_critic`: C51-style distributional values
- `twin_critic`: Twin Q-networks for SAC-style algorithms
- `world_model_critic`: Value function in latent space
- `hierarchical_critic`: Multi-level value decomposition
- `attention_critic`: Transformer-based value function

#### 2.3 Planning Systems
```python
class BasePlanner(nn.Module):
    """Plans actions using world models"""
    def plan(self,
             current_state: Tensor,
             dynamics_model: BaseDynamicsModel,
             value_function: BaseValueFunction,
             horizon: int) -> Distribution:
        """Plan actions given models"""
        pass

    @property
    def requires_differentiable_dynamics(self) -> bool:
        """Whether planner needs gradients through dynamics"""
        pass
```

**Registry of Planners**:
- `random_shooting`: Simple random action sampling
- `cem_planner`: Cross-Entropy Method planning
- `mpc_planner`: Model Predictive Control
- `mcts_planner`: Monte Carlo Tree Search (MuZero-style)
- `gradient_planner`: Gradient-based trajectory optimization
- `learned_planner`: Amortized planning via neural networks
- `hierarchical_planner`: Multi-level planning decomposition

### 3. Action Components (How agents decide what to do)

#### 3.1 Policy Heads
```python
class BasePolicyHead(nn.Module):
    """Converts representations to action distributions"""
    def forward(self, representation: Tensor, context: Dict[str, Any]) -> Distribution:
        """representation + context -> action_distribution"""
        pass

    def sample_actions(self, representation: Tensor, context: Dict[str, Any]) -> Tensor:
        """Sample actions for execution"""
        pass

    def action_log_prob(self, actions: Tensor, representation: Tensor, context: Dict[str, Any]) -> Tensor:
        """Compute log probability of actions"""
        pass
```

**Registry of Policy Heads**:
- `deterministic_mlp`: Simple deterministic policy
- `gaussian_mlp`: Gaussian policy for continuous control
- `categorical_mlp`: Discrete action policy
- `diffusion_policy`: Diffusion-based action generation
- `autoregressive_policy`: Sequential action prediction
- `mixture_policy`: Mixture of expert policies
- `hierarchical_policy`: Multi-level action decomposition
- `constrained_policy`: Safety-constrained action selection

**Diffusion Policy Implementation**:
```python
class DiffusionPolicyHead(BasePolicyHead):
    def __init__(self, backbone_type="unet", horizon=10, noise_steps=100):
        self.backbone = self._create_backbone(backbone_type)
        self.noise_scheduler = NoiseScheduler(noise_steps)
        self.horizon = horizon

    def forward(self, representation, context):
        return DiffusionDistribution(
            backbone=self.backbone,
            conditioning=representation,
            context=context,
            noise_scheduler=self.noise_scheduler
        )

    def sample_actions(self, representation, context):
        # Iterative denoising to generate action sequence
        noise = torch.randn(self.horizon, self.action_dim)
        for t in reversed(range(self.noise_steps)):
            conditioning = torch.cat([representation, context.get('goal', torch.zeros_like(representation))])
            noise = self.backbone(noise, t, conditioning)
        return noise
```

#### 3.2 Action Space Adapters
```python
class BaseActionAdapter(nn.Module):
    """Adapts between different action representations"""
    def adapt_to_environment(self, policy_actions: Tensor) -> Tensor:
        """policy output -> environment actions"""
        pass

    def adapt_from_environment(self, env_actions: Tensor) -> Tensor:
        """environment actions -> policy input"""
        pass
```

**Registry of Action Adapters**:
- `continuous_adapter`: Handles continuous action spaces
- `discrete_adapter`: Discretization/one-hot encoding
- `hybrid_adapter`: Mixed continuous-discrete actions
- `temporal_adapter`: Fixed-length sequences to variable-length
- `safety_adapter`: Adds safety constraints/filtering

### 4. Integration Components (How everything works together)

#### 4.1 Paradigm Templates
```python
class BaseParadigm(nn.Module):
    """High-level agent architectures"""
    def __init__(self,
                 encoder: BaseEncoder,
                 representation_learner: BaseRepresentationLearner,
                 policy_head: BasePolicyHead,
                 **kwargs):
        self.encoder = encoder
        self.representation_learner = representation_learner
        self.policy_head = policy_head

    def forward(self, observations: Dict[str, Tensor], context: Dict[str, Any]) -> Distribution:
        """Complete forward pass through agent"""
        pass

    def compute_loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """All learning objectives for this paradigm"""
        pass
```

**Registry of Paradigms**:

**Model-Free Paradigm**:
```python
class ModelFreeParadigm(BaseParadigm):
    def __init__(self, encoder, representation_learner, policy_head, value_function):
        super().__init__(encoder, representation_learner, policy_head)
        self.value_function = value_function

    def compute_loss(self, batch):
        features = self.encoder(batch['observations'])
        representations = self.representation_learner.encode(features)

        # Policy loss (PPO, SAC, etc.)
        action_dist = self.policy_head(representations, batch)
        policy_loss = -action_dist.log_prob(batch['actions']).mean()

        # Value loss
        values = self.value_function(representations)
        value_loss = F.mse_loss(values, batch['returns'])

        # Representation loss (if any)
        repr_loss = self.representation_learner.representation_loss(features)

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'representation_loss': repr_loss
        }
```

**World Model Paradigm**:
```python
class WorldModelParadigm(BaseParadigm):
    def __init__(self, encoder, representation_learner, dynamics_model,
                 policy_head, value_function, planner=None):
        super().__init__(encoder, representation_learner, policy_head)
        self.dynamics_model = dynamics_model
        self.value_function = value_function
        self.planner = planner

    def forward(self, observations, context):
        features = self.encoder(observations)
        state = self.representation_learner.encode(features)

        if self.planner:
            # Plan using world model
            action_dist = self.planner.plan(state, self.dynamics_model, self.value_function, horizon=10)
        else:
            # Direct policy prediction
            action_dist = self.policy_head(state, context)

        return action_dist

    def compute_loss(self, batch):
        # Encode observations
        features = self.encoder(batch['observations'])
        states = self.representation_learner.encode(features)

        # Reconstruction loss
        reconstructed = self.representation_learner.decode(states)
        reconstruction_loss = F.mse_loss(reconstructed, features)

        # Dynamics loss
        next_features = self.encoder(batch['next_observations'])
        next_states = self.representation_learner.encode(next_features)
        predicted_next_states = self.dynamics_model(states, batch['actions'])
        dynamics_loss = F.mse_loss(predicted_next_states.mean, next_states)

        # Policy loss (learning in imagination)
        imagined_trajectories = self.rollout_imagination(states, length=10)
        policy_loss = self.compute_policy_loss(imagined_trajectories)

        return {
            'reconstruction_loss': reconstruction_loss,
            'dynamics_loss': dynamics_loss,
            'policy_loss': policy_loss
        }
```

**VLA Paradigm**:
```python
class VLAParadigm(BaseParadigm):
    def __init__(self, vision_encoder, language_encoder, transformer, policy_head):
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.transformer = transformer
        self.policy_head = policy_head

    def forward(self, observations, context):
        # Multi-modal tokenization
        vision_tokens = self.vision_encoder(observations['image'])
        language_tokens = self.language_encoder(context['instruction'])

        # Process with transformer
        all_tokens = torch.cat([vision_tokens, language_tokens], dim=1)
        hidden_states = self.transformer(all_tokens)

        # Extract action-relevant representation
        action_representation = hidden_states[:, -1]  # Use last token

        # Generate actions
        return self.policy_head(action_representation, context)
```

---

## Hybrid Architectures: The Innovation Engine

### Hybrid 1: VLA + World Model Planning
```python
class VLAWorldModelHybrid(BaseParadigm):
    """VLA provides goals, world model plans to achieve them"""

    def __init__(self, vla_paradigm, world_model_paradigm):
        self.vla = vla_paradigm          # High-level reasoning
        self.world_model = world_model_paradigm  # Low-level planning

    def forward(self, observations, context):
        # VLA predicts high-level subgoals/plans
        high_level_plan = self.vla(observations, context)

        # Extract subgoal from VLA output
        if 'subgoal' in high_level_plan:
            subgoal = high_level_plan['subgoal']
        else:
            subgoal = self.extract_subgoal(high_level_plan)

        # World model plans actions to reach subgoal
        planning_context = {**context, 'goal': subgoal}
        return self.world_model(observations, planning_context)

    def compute_loss(self, batch):
        # Train both components
        vla_losses = self.vla.compute_loss(batch)
        wm_losses = self.world_model.compute_loss(batch)

        # Add alignment loss between VLA goals and achieved outcomes
        alignment_loss = self.compute_goal_alignment_loss(batch)

        return {
            **{f'vla_{k}': v for k, v in vla_losses.items()},
            **{f'wm_{k}': v for k, v in wm_losses.items()},
            'alignment_loss': alignment_loss
        }
```

### Hybrid 2: Multi-Scale Diffusion Policy
```python
class MultiScaleDiffusionHybrid(BaseParadigm):
    """Hierarchical diffusion: coarse motions + fine corrections"""

    def __init__(self, encoder, coarse_diffusion_head, fine_diffusion_head):
        self.encoder = encoder
        self.coarse_diffusion = coarse_diffusion_head    # Long horizon, low frequency
        self.fine_diffusion = fine_diffusion_head        # Short horizon, high frequency

    def forward(self, observations, context):
        features = self.encoder(observations)

        # Generate coarse trajectory
        coarse_actions = self.coarse_diffusion.sample_actions(
            features, {**context, 'horizon': 50, 'frequency': 'low'}
        )

        # Generate fine corrections conditioned on coarse plan
        fine_context = {**context, 'coarse_plan': coarse_actions[:10]}
        fine_actions = self.fine_diffusion.sample_actions(features, fine_context)

        # Combine hierarchically
        return self.combine_hierarchical_actions(coarse_actions, fine_actions)
```

### Hybrid 3: Ensemble Policy Head
```python
class EnsemblePolicyHybrid(BaseParadigm):
    """Multiple policy heads with learned weighting"""

    def __init__(self, encoder, policy_heads, weighting_network):
        self.encoder = encoder
        self.policy_heads = nn.ModuleDict(policy_heads)  # {'diffusion': ..., 'gaussian': ..., 'categorical': ...}
        self.weighting_network = weighting_network

    def forward(self, observations, context):
        features = self.encoder(observations)

        # Get outputs from all policy heads
        policy_outputs = {}
        for name, head in self.policy_heads.items():
            policy_outputs[name] = head(features, context)

        # Learn weights based on context
        weights = self.weighting_network(features, context)
        weights = torch.softmax(weights, dim=-1)

        # Weighted combination of policies
        return MixtureDistribution(policy_outputs, weights)
```

### Hybrid 4: Representation Transfer
```python
class RepresentationTransferHybrid(BaseParadigm):
    """Train representations with one paradigm, use with another"""

    def __init__(self, source_paradigm, target_policy_head, transfer_adapter):
        self.source_paradigm = source_paradigm  # e.g., world model
        self.target_policy_head = target_policy_head  # e.g., diffusion policy
        self.transfer_adapter = transfer_adapter  # learns mapping between representations

        # Freeze source representations after pretraining
        for param in self.source_paradigm.parameters():
            param.requires_grad = False

    def forward(self, observations, context):
        # Get representations from source paradigm
        with torch.no_grad():
            source_features = self.source_paradigm.encoder(observations)
            source_repr = self.source_paradigm.representation_learner.encode(source_features)

        # Adapt representations for target paradigm
        target_repr = self.transfer_adapter(source_repr)

        # Use target policy head
        return self.target_policy_head(target_repr, context)
```

---

## Configuration System: Making It All Work

### Universal Configuration Schema
```yaml
agent:
  paradigm: "custom_hybrid"  # or "model_free", "world_model", "vla"

  # Perception pipeline
  encoder:
    type: "dino_v2"
    config:
      model_size: "base"
      frozen: true
      fine_tune_layers: 2

  representation_learner:
    type: "vae"
    config:
      latent_dim: 512
      beta: 1.0

  # Reasoning components (optional, depends on paradigm)
  dynamics_model:
    type: "dreamer_rssm"
    config:
      hidden_dim: 512
      sequence_length: 50

  value_function:
    type: "distributional_critic"
    config:
      num_atoms: 51
      v_min: -10
      v_max: 10

  planner:
    type: "cem_planner"
    config:
      num_samples: 1000
      num_iterations: 10
      horizon: 15

  # Action generation
  policy_head:
    type: "diffusion_policy"
    config:
      backbone: "unet"
      horizon: 10
      noise_steps: 100
      noise_schedule: "cosine"

  action_adapter:
    type: "continuous_adapter"
    config:
      action_bounds: [-1.0, 1.0]
      clip_actions: true

# Hybrid-specific configurations
hybrid_config:
  type: "vla_world_model"
  vla_weight: 0.7
  world_model_weight: 0.3
  alignment_loss_weight: 0.1
```

### Component Factory System
```python
class ComponentFactory:
    """Creates components from configuration"""

    @staticmethod
    def create_agent(config: Dict) -> BaseParadigm:
        # Create all components
        encoder = ComponentFactory.create_encoder(config['encoder'])
        repr_learner = ComponentFactory.create_representation_learner(config.get('representation_learner'))

        # Paradigm-specific creation
        if config['paradigm'] == 'model_free':
            value_fn = ComponentFactory.create_value_function(config['value_function'])
            policy_head = ComponentFactory.create_policy_head(config['policy_head'])
            return ModelFreeParadigm(encoder, repr_learner, policy_head, value_fn)

        elif config['paradigm'] == 'world_model':
            dynamics = ComponentFactory.create_dynamics_model(config['dynamics_model'])
            value_fn = ComponentFactory.create_value_function(config['value_function'])
            policy_head = ComponentFactory.create_policy_head(config['policy_head'])
            planner = ComponentFactory.create_planner(config.get('planner'))
            return WorldModelParadigm(encoder, repr_learner, dynamics, policy_head, value_fn, planner)

        elif config['paradigm'] == 'vla':
            lang_encoder = ComponentFactory.create_language_encoder(config['language_encoder'])
            transformer = ComponentFactory.create_transformer(config['transformer'])
            policy_head = ComponentFactory.create_policy_head(config['policy_head'])
            return VLAParadigm(encoder, lang_encoder, transformer, policy_head)

        elif config['paradigm'] == 'custom_hybrid':
            return ComponentFactory.create_hybrid(config['hybrid_config'])

    @staticmethod
    def create_hybrid(hybrid_config: Dict) -> BaseParadigm:
        hybrid_type = hybrid_config['type']

        if hybrid_type == 'vla_world_model':
            vla = ComponentFactory.create_agent({...})  # VLA config
            world_model = ComponentFactory.create_agent({...})  # World model config
            return VLAWorldModelHybrid(vla, world_model)

        elif hybrid_type == 'multi_scale_diffusion':
            # Create multi-scale diffusion hybrid
            pass

        # Add more hybrid types as needed
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
1. **Base Interfaces**: Implement all abstract base classes
2. **Component Registry**: Dynamic loading system for all components
3. **Configuration System**: YAML-based component specification
4. **Factory Pattern**: Automatic component creation from configs

### Phase 2: Basic Components (Weeks 3-4)
1. **Encoders**: IMPALA CNN, ResNet, DINO v2
2. **Policy Heads**: Gaussian MLP, Categorical MLP, Diffusion Policy
3. **Simple Paradigms**: Model-free (PPO-style), Basic World Model

### Phase 3: Advanced Components (Weeks 5-6)
1. **Dynamics Models**: RSSM, Transformer, Ensemble
2. **Value Functions**: Distributional, Twin Critics
3. **Planners**: CEM, MPC, Learned Planning
4. **VLA Components**: Language encoders, Transformers

### Phase 4: Hybrid Systems (Weeks 7-8)
1. **Multi-Paradigm Combinations**: VLA + World Model, Ensemble Policies
2. **Transfer Learning**: Representation sharing across paradigms
3. **Hierarchical Agents**: Multi-scale reasoning and control

### Phase 5: Optimization & Research Tools (Weeks 9-10)
1. **Experiment Management**: Hyperparameter sweeps, A/B testing
2. **Analysis Tools**: Component ablation studies, performance profiling
3. **Research Workflows**: Rapid prototyping, result visualization

---

## Usage Examples: From Simple to Complex

### Example 1: Simple PPO Agent
```yaml
agent:
  paradigm: "model_free"
  encoder: {type: "impala_cnn"}
  policy_head: {type: "gaussian_mlp", config: {hidden_dims: [256, 256]}}
  value_function: {type: "mlp_critic", config: {hidden_dims: [256, 256]}}
```

### Example 2: Dreamer-style World Model
```yaml
agent:
  paradigm: "world_model"
  encoder: {type: "impala_cnn"}
  representation_learner: {type: "vae", config: {latent_dim: 512}}
  dynamics_model: {type: "dreamer_rssm", config: {hidden_dim: 512}}
  policy_head: {type: "gaussian_mlp"}
  value_function: {type: "mlp_critic"}
```

### Example 3: Diffusion Policy with Pretrained Vision
```yaml
agent:
  paradigm: "model_free"
  encoder: {type: "dino_v2", config: {frozen: true}}
  policy_head: {type: "diffusion_policy", config: {horizon: 10, noise_steps: 100}}
  value_function: {type: "mlp_critic"}
```

### Example 4: VLA with Planning
```yaml
agent:
  paradigm: "custom_hybrid"
  hybrid_config:
    type: "vla_world_model"
    vla:
      vision_encoder: {type: "clip_vision"}
      language_encoder: {type: "gpt2_small"}
      transformer: {type: "gpt_transformer"}
      policy_head: {type: "categorical_mlp"}
    world_model:
      encoder: {type: "clip_vision"}  # Shared with VLA
      dynamics_model: {type: "transformer_dynamics"}
      planner: {type: "mpc_planner"}
      policy_head: {type: "gaussian_mlp"}
```

---

## Benefits of This Architecture

### For Researchers
- **Rapid Prototyping**: Test new ideas by swapping components
- **Fair Comparison**: Same infrastructure for different approaches
- **Innovation**: Easy to combine ideas from different paradigms
- **Reproducibility**: Deterministic configs ensure repeatable experiments

### For the Field
- **Standardization**: Common interfaces enable collaboration
- **Modularity**: Components can be developed independently
- **Extensibility**: New paradigms integrate seamlessly
- **Knowledge Transfer**: Insights from one approach benefit others

### For You Specifically
- **Learning**: Understand concepts by seeing them work together
- **Research**: Build on state-of-the-art without reimplementation
- **Efficiency**: Focus on ideas, not infrastructure
- **Claude Code Synergy**: Specify what you want, not how to implement it

This architecture transforms your repository from a collection of algorithms into a research laboratory where any combination of ideas can be tested systematically. The modular design ensures that as the field evolves, your platform evolves with it, always supporting the latest paradigms while maintaining backward compatibility with proven approaches.