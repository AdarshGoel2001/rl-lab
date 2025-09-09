Here’s how vision works today vs how it should, plus a clean way to wire a shared encoder with minimal disruption to
  PPO and your trainer.

  Current State

  - Invocation point:
      - Trainer builds two separate networks from config: networks['actor'] and networks['critic'].
      - PPO calls them independently: actor(observation) for logits/means; critic(observation) for values (see
  PPOAlgorithm.act, get_action_and_value, evaluate_actions).
  - Vision today:
      - If you choose CNNs (e.g., nature_cnn) for both actor and critic, each module contains its own conv trunk + FC
  head. There is no shared encoder; vision weights are duplicated.
      - CNNs expect image inputs in BHWC but convert internally to BCHW.
      - Atari-specific preprocessing happens inside AtariEnvironment; Gym/VectorizedGym currently pass raw obs without
  a general transform system.
  - Checkpointing:
      - BaseAlgorithm saves all networks in self.networks via state_dict(). So if actor/critic are separate CNNs, both
  sets of vision weights are saved.

  Target Design

  - Shared vision encoder:
      - One backbone encodes obs → features once.
      - Two lightweight heads consume features:
      - Actor head: outputs logits (discrete) or mean + log_std (continuous).
      - Critic head: outputs scalar value.
  - End-to-end training: PPO loss backpropagates through heads and the shared encoder; the encoder learns.
  - Invocation point:
      - Keep PPO unchanged: it still calls networks['actor'](obs) and networks['critic'](obs).
      - Both actor and critic modules internally call the same shared encoder instance. This preserves PPO/trainer
  interfaces.
  - Inputs and flow:
      - Env/Transforms: observation → BHWC float32 (e.g., 84x84x4 in [0,1]).
      - Shared encoder: BHWC → features (torch.tensor).
      - Actor head MLP: features → logits or mean/log_std.
      - Critic head MLP: features → value.
      - PPO sampling/distributions remain as implemented.

  Where to place it

  - Add a dedicated “vision backbone” network class that returns features only (no heads).
  - Add “actor head” and “critic head” that take feature_dim and produce outputs.
  - Wire them together in the Trainer so the head modules both reference the same encoder instance.

  Minimal Integration Plan (no PPO changes)

  - New networks:
      - VisionBackbone (e.g., nature_cnn_backbone): returns features with get_feature_dim().
      - ActorHeadMLP: input_dim=feature_dim, output_dim=action_dim (discrete) or 2*action_dim (continuous).
      - CriticHeadMLP: input_dim=feature_dim, output_dim=1.
      - SharedVisionActor: wraps a shared backbone instance and an ActorHeadMLP. Its forward(obs) =
  head(backbone(obs)).
      - SharedVisionCritic: same pattern with CriticHeadMLP.
  - Trainer wiring:
      - If config specifies a shared vision block, construct the backbone once.
      - Construct SharedVisionActor(backbone=shared_backbone, ...) and
  SharedVisionCritic(backbone=shared_backbone, ...).
      - Register them as networks['actor'] and networks['critic'].
  - Config example:
      - network:
      - backbone:
        - type: nature_cnn_backbone
        - channels: [32,64,64]
      - actor:
        - type: shared_vision_actor_head
        - hidden_dims: [512]
      - critic:
        - type: shared_vision_critic_head
        - hidden_dims: [512]
  - Shapes:
      - Env emits (B,H,W,C). Backbone permutes to (B,C,H,W) internally, outputs (B, F).
      - Heads take (B, F) and output (B, A) or (B, 2A) (actor), (B, 1) (critic).

  Alternative (combined module)

  - A single ActorCriticShared module with forward_policy(obs) and forward_value(obs), or forward(obs) -> (logits,
  value). Then update PPO to use this single module.
  - Pros: centralizes the feature pass. Cons: requires PPO changes and deviates from your current separation.

  Checkpointing

  - Nothing extra needed. Because SharedVisionActor and SharedVisionCritic both hold references to the same backbone
  submodule, BaseAlgorithm.save_checkpoint() captures the shared weights exactly once (PyTorch handles shared
  submodules correctly). On load, state dict rebinds into the same shared structure.

  Putting it together with transforms

  - Env side (now/future): apply observation transforms (grayscale/resize/scale/frame stack) so the network always
  sees standardized BHWC float32 input.
  - Networks side: backbone converts BHWC→BCHW and encodes.
  - PPO: unchanged, still calls actor/critic with processed tensors.

  Recommended next steps

  - Add a small vision backbone class (start by refactoring NatureCNN into NatureCNNBackbone + NatureCNNHead so the
  backbone returns features and a separate FC maps features→output).
  - Add SharedVisionActor/SharedVisionCritic thin wrappers that reuse the same backbone instance.
  - Update Trainer to:
      - Detect network.backbone config; instantiate it once.
      - Pass the instance into actor and critic head constructors (your registry can accept the module instance in the
  config dict).
  - Keep PPO untouched; it will automatically train encoder+heads end-to-end.
  - Verify with a pixel env: confirm shapes, that both heads share parameters (parameter IDs overlap for the
  backbone), and that checkpoints load.