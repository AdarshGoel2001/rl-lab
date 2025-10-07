# World Model Gradient Flow Diagrams

## Diagram 1: Complete Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         WORLD MODEL PARADIGM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────────┘

                                REAL DATA INPUT
                                      │
                                      ▼
                            ┌──────────────────┐
                            │   Observations   │
                            │  (raw env data)  │
                            └────────┬─────────┘
                                     │
                                     │ [84x84x3] or [4,]
                                     ▼
                            ┌──────────────────┐
                            │     ENCODER      │◄─────────┐
                            │  (CNN or MLP)    │          │
                            └────────┬─────────┘          │
                                     │                    │
                                     │ features [256]     │
                                     ▼                    │
                    ┌────────────────────────────┐       │
                    │  REPRESENTATION LEARNER    │       │ Reconstruction
                    │    (Autoencoder)          │       │ gradient
                    │  .encode() / .decode()    │       │
                    └───────────┬────────────────┘       │
                                │                        │
                                │ states [64]            │
                                ▼                        │
                    ┌───────────────────────┐            │
                    │   LEARNED STATES      │            │
                    │   (compressed repr)   │            │
                    └───┬───┬───┬───┬───┬───┘            │
                        │   │   │   │   │                │
        ┌───────────────┤   │   │   │   └─────────────┐  │
        │               │   │   │   └─────────┐       │  │
        │               │   │   └─────┐       │       │  │
        │               │   └───┐     │       │       │  │
        ▼               ▼       ▼     ▼       ▼       ▼  │
   ┌─────────┐    ┌─────────┐ ┌──────────┐ ┌──────┐ ┌──────────┐
   │DYNAMICS │    │ REWARD  │ │CONTINUE  │ │POLICY│ │  VALUE   │
   │  MODEL  │    │PREDICTOR│ │PREDICTOR │ │ HEAD │ │FUNCTION  │
   └────┬────┘    └────┬────┘ └────┬─────┘ └───┬──┘ └────┬─────┘
        │              │           │            │         │
        │next_s        │reward     │continue    │action   │value
        │dist          │dist       │dist        │dist     │
        ▼              ▼           ▼            ▼         ▼
   ┌──────────────────────────────────────────────────────────┐
   │                  IMAGINATION ROLLOUT                      │
   │  (15 steps into the future using learned world model)   │
   └─────────────────────┬────────────────────────────────────┘
                         │
                         │ imagined trajectory
                         ▼
        ┌────────────────────────────────────────┐
        │  COMPUTE RETURNS (λ-returns)           │
        │  return_t = r_t + γ·c_t·[(1-λ)V + λR] │
        └────────┬──────────────────┬────────────┘
                 │                  │
                 │                  │
        actor_loss             critic_loss
        = -mean(returns)       = MSE(V, returns)
                 │                  │
                 ▼                  ▼
        ┌──────────────┐   ┌──────────────┐
        │ACTOR OPTIMIZE│   │CRITIC OPTIMIZE│
        │  (policy)    │   │ (value fn)   │
        └──────────────┘   └──────────────┘


LEGEND:
  ┌────┐  = Neural Network Component
  │    │
  └────┘

  ───▶   = Data Flow (forward pass)

  ◄────  = Gradient Flow (backward pass)
```

---

## Diagram 2: Loss Computation and Gradient Flow

```
═══════════════════════════════════════════════════════════════════════════════
                        LOSS COMPUTATION PIPELINE
═══════════════════════════════════════════════════════════════════════════════

PHASE 1: WORLD MODEL LOSSES (learn to predict environment)
─────────────────────────────────────────────────────────────────────────────

Real Batch Data:
  observations: [B, obs_dim]
  actions: [B, act_dim]
  rewards: [B,]
  next_observations: [B, obs_dim]
  dones: [B,]

          │
          │ (process batch)
          ▼
  ┌───────────────────┐
  │  Encode obs & next│
  │      obs          │
  └─────────┬─────────┘
            │
            ├──────────────────────────────────┐
            │                                  │
            ▼                                  ▼
     ┌──────────┐                      ┌──────────┐
     │  states  │                      │next_states│
     └─────┬────┘                      └─────┬────┘
           │                                 │
           │                                 │
    ┌──────┴──────────────────────────┬─────┴─────┬──────────────┐
    │                                 │           │              │
    │                                 │           │              │
    ▼                                 ▼           ▼              ▼
┌─────────────────┐          ┌──────────────┐ ┌────────────┐ ┌────────────┐
│ DYNAMICS LOSS   │          │ REWARD LOSS  │ │CONTINUE LS │ │DECODER LOSS│
│                 │          │              │ │            │ │            │
│ dynamics_model  │          │reward_pred   │ │continue_pr │ │decoder     │
│  (s, a)         │          │  (s)         │ │  (s)       │ │  (s)       │
│    ↓            │          │    ↓         │ │    ↓       │ │    ↓       │
│ predicted_s'    │          │ predicted_r  │ │predicted_c │ │reconst_obs │
│    ↓            │          │    ↓         │ │    ↓       │ │    ↓       │
│ MSE(pred, true) │          │ NLL(pred, r) │ │BCE(pred, c)│ │MSE(rec,obs)│
│                 │          │              │ │            │ │            │
│   ↓ 0.15        │          │   ↓ 1.2     │ │   ↓ 0.08   │ │   ↓ 0.05   │
└────┬────────────┘          └──────┬───────┘ └─────┬──────┘ └─────┬──────┘
     │                              │               │              │
     └──────────────┬───────────────┴───────────────┴──────────────┘
                    │
                    │ SUM
                    ▼
         ┌─────────────────────┐
         │ WORLD_MODEL_LOSS    │
         │ = dynamics + reward │
         │   + continue + dec  │
         │                     │
         │   = 1.48            │
         └──────────┬──────────┘
                    │
                    │ .backward(retain_graph=True)
                    ▼
         ┌─────────────────────┐
         │  GRADIENTS FLOW TO: │
         │  - encoder          │
         │  - repr_learner     │
         │  - dynamics_model   │
         │  - reward_predictor │
         │  - continue_pred    │
         │  - decoder          │
         └──────────┬──────────┘
                    │
                    │ clip_grad_norm(1.0)
                    ▼
         ┌─────────────────────┐
         │world_model_optimizer│
         │      .step()        │
         └─────────────────────┘


─────────────────────────────────────────────────────────────────────────────
PHASE 2: ACTOR LOSS (learn to choose good actions)
─────────────────────────────────────────────────────────────────────────────

Take states from real data
         │
         │ [B, state_dim]
         ▼
┌────────────────────────────────────────────────────┐
│  IMAGINATION ROLLOUT (with_grad=True)              │
│                                                    │
│  for t in range(15):  # Imagine 15 steps ahead    │
│      action ~ policy(state_t)                     │
│      state_{t+1} ~ dynamics(state_t, action)      │
│      reward_t = reward_pred(state_t).mean         │
│      continue_t = continue_pred(state_t).probs    │
│      value_t = value_fn(state_t)                  │
│                                                    │
│  Returns:                                          │
│    states: [B, 15, state_dim]                     │
│    actions: [B, 15, action_dim]                   │
│    rewards: [B, 15]                               │
│    values: [B, 15]                                │
│    continues: [B, 15]                             │
└──────────────────┬─────────────────────────────────┘
                   │
                   │ (use rewards, values, continues)
                   ▼
         ┌─────────────────────┐
         │ COMPUTE λ-RETURNS   │
         │                     │
         │ for t reversed:     │
         │   R_t = r_t + γ·c_t │
         │     ·[(1-λ)V_t+1    │
         │       + λ·R_t+1]    │
         │                     │
         │ Returns: [B, 15]    │
         └──────────┬──────────┘
                    │
                    ├──────────────┐
                    │              │
                    ▼              ▼
         ┌──────────────┐  ┌──────────────┐
         │ ACTOR LOSS   │  │ ENTROPY      │
         │              │  │              │
         │ -mean(R)     │  │ -Σp log(p)   │
         │              │  │              │
         │  = -12.5     │  │  = 1.2       │
         └──────┬───────┘  └──────┬───────┘
                │                 │
                │ + entropy_coef × entropy
                ▼
         ┌──────────────────┐
         │ TOTAL ACTOR LOSS │
         │ = -12.5 - 0.01×1.2│
         │ = -12.512        │
         └──────┬───────────┘
                │
                │ .backward(retain_graph=True)
                ▼
         ┌──────────────────────────────────┐
         │  GRADIENTS FLOW THROUGH:         │
         │  returns ← rewards ← reward_pred │
         │         ↖           ↗            │
         │          dynamics_model          │
         │                ↑                 │
         │            actions               │
         │                ↑                 │
         │           policy_head ← UPDATE   │
         └──────────────┬───────────────────┘
                        │
                        │ clip_grad_norm(1.0)
                        ▼
         ┌──────────────────────┐
         │  actor_optimizer     │
         │      .step()         │
         └──────────────────────┘


NOTE: Gradients flow THROUGH dynamics/reward (not updating them),
      and TO policy (updating it).


─────────────────────────────────────────────────────────────────────────────
PHASE 3: CRITIC LOSS (learn to evaluate states)
─────────────────────────────────────────────────────────────────────────────

From imagination rollout:
  imagined_states: [B, 15, state_dim]
  computed_returns: [B, 15]  ← .detach() [no grad!]

         │
         │ flatten
         ▼
  states: [B×15, state_dim]
  returns: [B×15]  [DETACHED]
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
  ┌──────────┐      ┌──────────┐
  │value_fn  │      │ returns  │
  │ (states) │      │ (target) │
  └─────┬────┘      └─────┬────┘
        │                 │
        │ predicted       │ target (no grad)
        ▼                 ▼
  ┌─────────────────────────────┐
  │      CRITIC LOSS            │
  │                             │
  │  MSE(predicted, target)     │
  │                             │
  │  = mean((V(s) - R)²)        │
  │                             │
  │  = 2.1                      │
  └───────────┬─────────────────┘
              │
              │ .backward()
              ▼
  ┌───────────────────────────┐
  │  GRADIENTS FLOW TO:       │
  │  - value_function ONLY    │
  │                           │
  │  (returns.detach() blocks │
  │   gradient to policy/WM)  │
  └───────────┬───────────────┘
              │
              │ clip_grad_norm(1.0)
              ▼
  ┌───────────────────────────┐
  │  critic_optimizer         │
  │      .step()              │
  └───────────────────────────┘
```

---

## Diagram 3: Three Optimizer Dance (Critical Understanding)

```
═══════════════════════════════════════════════════════════════════════
                    THREE SEPARATE BACKWARD PASSES
═══════════════════════════════════════════════════════════════════════

TIME: Start of update() call
─────────────────────────────────────────────────────────────────────────
                    ┌─────────────────────┐
                    │  COMPUTE ALL LOSSES │
                    │  (single forward)   │
                    └──────────┬──────────┘
                               │
                    losses = { world_model_loss: 1.48,
                              actor_loss: -12.5,
                              critic_loss: 2.1 }
                               │
                               ▼

╔════════════════════════════════════════════════════════════════════╗
║  STEP 1: WORLD MODEL BACKWARD                                      ║
╚════════════════════════════════════════════════════════════════════╝

world_model_loss.backward(retain_graph=True)
         │
         └──────► Computes gradients for:
                  - encoder.parameters()
                  - repr_learner.parameters()
                  - dynamics_model.parameters()
                  - reward_predictor.parameters()
                  - continue_predictor.parameters()
                  - decoder.parameters()

         ┌────────────────────────────────────┐
         │ SAVE THESE GRADIENTS!              │
         │ (will be corrupted by next step)   │
         └────────────────────────────────────┘
                  │
                  │ saved_wm_grads = [p.grad.clone() for p in wm_params]
                  ▼

╔════════════════════════════════════════════════════════════════════╗
║  STEP 2: ACTOR BACKWARD                                            ║
╚════════════════════════════════════════════════════════════════════╝

actor_loss.backward(retain_graph=True)
         │
         └──────► Computes gradients for:
                  - policy_head.parameters() ← WANT THIS
                  - dynamics_model.parameters() ← DON'T WANT (side effect)
                  - reward_predictor.parameters() ← DON'T WANT (side effect)

         ┌────────────────────────────────────┐
         │ PROBLEM: dynamics & reward now     │
         │ have gradients from BOTH world     │
         │ model loss AND actor loss!         │
         │                                    │
         │ We only want world model to be     │
         │ updated based on world model loss. │
         └────────────────────────────────────┘
                  │
                  │ SOLUTION: Restore saved gradients
                  ▼
         ┌────────────────────────────────────┐
         │ for param, saved_grad in zip(...): │
         │     param.grad = saved_grad        │
         └────────────────────────────────────┘
                  │
                  ▼

╔════════════════════════════════════════════════════════════════════╗
║  STEP 3: CRITIC BACKWARD                                           ║
╚════════════════════════════════════════════════════════════════════╝

critic_loss.backward()  ← No retain_graph (last one)
         │
         └──────► Computes gradients for:
                  - value_function.parameters() ← ONLY THIS

         ┌────────────────────────────────────┐
         │ returns.detach() prevents gradient │
         │ from flowing to world model/policy │
         └────────────────────────────────────┘
                  │
                  ▼

╔════════════════════════════════════════════════════════════════════╗
║  STEP 4: CLIP AND STEP ALL OPTIMIZERS                              ║
╚════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────┐
│ clip_grad_norm(wm_params, 1.0)   │
│ world_model_optimizer.step()     │
└──────────────────────────────────┘
         │ Updates: encoder, repr, dynamics, reward, continue, decoder
         ▼
┌──────────────────────────────────┐
│ clip_grad_norm(actor_params, 1.0)│
│ actor_optimizer.step()           │
└──────────────────────────────────┘
         │ Updates: policy_head
         ▼
┌──────────────────────────────────┐
│ clip_grad_norm(critic_params, 1.0)│
│ critic_optimizer.step()          │
└──────────────────────────────────┘
         │ Updates: value_function
         ▼
    ┌────────────┐
    │    DONE    │
    └────────────┘


KEY INSIGHT:
────────────
Each component is updated based on its OWN loss:
  - World model ← world_model_loss (prediction error)
  - Actor ← actor_loss (imagined returns)
  - Critic ← critic_loss (value estimation error)

Even though they share the computation graph (actor uses world model),
they are optimized independently.
```

---

## Diagram 4: Sequence Processing and Context Flow

```
═══════════════════════════════════════════════════════════════════════
         SEQUENCE-AWARE PROCESSING (when enabled)
═══════════════════════════════════════════════════════════════════════

Input batch can be:
  1. Flat: [B, obs_dim]  ← Independent transitions
  2. Sequence: [B, T, obs_dim]  ← Temporally related

Your friend added support for BOTH!

FLAT MODE (sequence_mode = False):
────────────────────────────────────────────────────────────────────
observations: [64, 4]  ← 64 independent CartPole states
         │
         ▼
   encoder(obs)
         │
         ▼
   repr_learner.encode(features)
         │
         ▼
   states: [64, state_dim]
         │
         └──► Each component processes independently


SEQUENCE MODE (sequence_mode = True):
────────────────────────────────────────────────────────────────────
observations: [16, 50, 4]  ← 16 trajectories of 50 steps each
         │
         │ Flatten for encoder
         ▼
observations_flat: [800, 4]  ← Process all at once
         │
         ▼
   encoder(obs_flat)
         │
         ▼
features_flat: [800, feature_dim]
         │
         │ Reshape back to sequence
         ▼
features_seq: [16, 50, feature_dim]
         │
         │ Build sequence_context dict
         ▼
┌────────────────────────────────────────────────┐
│  sequence_context = {                          │
│    'sequence_length': 50,                      │
│    'batch_size': 16,                           │
│    'features': [16, 50, feat_dim],             │
│    'states': [16, 50, state_dim],              │
│    'actions': [16, 50, action_dim],            │
│    'rewards': [16, 50],                        │
│    'continues': [16, 50],                      │
│    'causal_mask': [1, 50, 50] (optional),      │
│    'padding_mask': [16, 50] (optional)         │
│  }                                             │
└────────────┬───────────────────────────────────┘
             │
             │ Attach context to all components
             ▼
┌─────────────────────────────────────────────────────┐
│  Each component can access context:                 │
│                                                     │
│  dynamics_model.sequence_context ← context dict    │
│  reward_predictor.sequence_context ← context dict  │
│  policy_head.sequence_context ← context dict       │
│  etc.                                               │
└─────────────────────────────────────────────────────┘


WHY SEQUENCE CONTEXT?
────────────────────────────────────────────────────────────────────
Future world models (like RSSM, Transformers) need:
  - Temporal information (what came before?)
  - Causal masks (don't look at future)
  - Padding masks (ignore done states)

Current deterministic MLP ignores context, but infrastructure is ready!

Example: Transformer Dynamics
────────────────────────────────────────────────────────────────────
class TransformerDynamics:
    def forward(self, state, action, context=None):
        if context and 'states' in context:
            # Use full sequence for attention
            state_sequence = context['states']  # [B, T, dim]
            causal_mask = context.get('causal_mask')  # [1, T, T]

            # Self-attention over past states
            attended = self.attention(
                state_sequence,
                mask=causal_mask  # Can't look at future
            )

            # Predict from attended context
            next_state = self.predict(attended, action)
        else:
            # Fallback to single-step
            next_state = self.predict(state, action)

        return Normal(next_state, self.std)
```

---

## Diagram 5: Warmup Schedule and Learning Phases

```
═══════════════════════════════════════════════════════════════════════
                    TRAINING SCHEDULE OVER TIME
═══════════════════════════════════════════════════════════════════════

Environment Steps:
0           5000         10000        50000       100000
│────────────│─────────────│───────────│────────────│
│  PHASE 1   │   PHASE 2   │         PHASE 3        │
│  WARMUP    │ TRANSITION  │      FULL TRAINING     │
│            │             │                        │


PHASE 1: WARMUP (steps 0 - 5000)
────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────┐
│  LEARN WORLD MODEL ONLY                                         │
│                                                                 │
│  World Model Updates: YES (1 per batch)                        │
│  Actor Updates: NO (forced to 0)                               │
│  Critic Updates: NO (forced to 0)                              │
│                                                                 │
│  Losses being optimized:                                        │
│    - dynamics_loss ←  Learn environment dynamics                │
│    - reward_loss ←  Learn reward structure                      │
│    - continue_loss ←  Learn termination conditions              │
│    - decoder_loss ←  Learn state representation                 │
│                                                                 │
│  Policy behavior: Random exploration                            │
│                                                                 │
│  WHY? Don't train policy on bad world model predictions!       │
└─────────────────────────────────────────────────────────────────┘

Metrics to monitor:
  dynamics_loss: 5.0 → 1.2 → 0.3 ✓ (decreasing)
  reward_mae: 2.0 → 0.8 → 0.2 ✓ (decreasing)
  continue_acc: 50% → 75% → 90% ✓ (improving)


PHASE 2: TRANSITION (steps 5000 - 10000)
────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────┐
│  BEGIN POLICY LEARNING                                          │
│                                                                 │
│  World Model Updates: YES (1 per batch)                        │
│  Actor Updates: YES (1 per batch) ← NOW ENABLED                │
│  Critic Updates: YES (1 per batch) ← NOW ENABLED               │
│                                                                 │
│  critic_real_mix: 1.0 → 0.5 → 0.0                              │
│    (gradually trust imagined returns more)                     │
│                                                                 │
│  Losses being optimized:                                        │
│    - All world model losses (continues)                        │
│    - actor_loss ←  NEW: Learn from imagination                 │
│    - critic_loss ←  NEW: Learn to evaluate states              │
│                                                                 │
│  WHY? World model quality sufficient, start learning policy    │
└─────────────────────────────────────────────────────────────────┘

Metrics to monitor:
  mean_imagined_return: Slowly increasing
  actor_grad_norm: Should be non-zero and stable
  entropy: High (still exploring)


PHASE 3: FULL TRAINING (steps 10000+)
────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────┐
│  JOINT OPTIMIZATION                                             │
│                                                                 │
│  World Model Updates: YES (can increase to 2-4)                │
│  Actor Updates: YES (1-2 per batch)                            │
│  Critic Updates: YES (1-2 per batch)                           │
│                                                                 │
│  All losses being optimized simultaneously                      │
│                                                                 │
│  World model continues to improve as policy explores           │
│  Policy exploits learned world model                           │
│  Critic provides accurate value estimates                      │
│                                                                 │
│  WHY? Co-evolution: better world model → better policy         │
│                     better policy → better data → better WM    │
└─────────────────────────────────────────────────────────────────┘

Metrics to monitor:
  dynamics_loss: Should stay low (~0.1)
  mean_imagined_return: Steadily increasing
  eval_return: Actual environment performance improving
  entropy: Gradually decreasing (exploitation)


OPTIONAL: ADVANCED SCHEDULES
────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────┐
│  MULTI-UPDATE SCHEDULE                                          │
│                                                                 │
│  Early training (steps < 50k):                                 │
│    wm_updates = 2  ← World model needs more updates            │
│    actor_updates = 1                                           │
│    critic_updates = 1                                          │
│                                                                 │
│  Late training (steps > 50k):                                  │
│    wm_updates = 1  ← World model stable                        │
│    actor_updates = 2  ← Focus on policy improvement            │
│    critic_updates = 2                                          │
└─────────────────────────────────────────────────────────────────┘

This is like curriculum learning for the agent!
```

---

## Diagram 6: Common Failure Modes and Their Signatures

```
═══════════════════════════════════════════════════════════════════════
                    DEBUGGING: WHAT WENT WRONG?
═══════════════════════════════════════════════════════════════════════

FAILURE 1: World Model Not Learning
────────────────────────────────────────────────────────────────────
Symptoms:
  dynamics_loss: 5.0 → 5.0 → 5.0 → 5.0 (FLAT)
  reward_mae: 2.0 → 2.0 → 2.0 (FLAT)
  world_model_grad_norm: 0.0001 (TINY)

Diagnosis: Vanishing gradients or learning rate too low

                     observations
                          │
                          ▼
                     ┌─────────┐
                     │ ENCODER │ ← Gradient: 0.0001
                     └────┬────┘
                          │
                     [Problem: No learning signal reaching encoder]
                          │
                          ▼
                     ┌──────────┐
                     │  STATES  │ ← All similar values
                     └──────────┘

Solutions:
  1. Increase world_model_lr: 1e-4 → 1e-3
  2. Check gradient flow: Print grad norms for each layer
  3. Verify data normalization: obs should be ~ N(0,1)
  4. Initialize weights properly: Use orthogonal init


FAILURE 2: Policy Learns Wrong Behavior
────────────────────────────────────────────────────────────────────
Symptoms:
  mean_imagined_return: 10 → 50 → 100 (INCREASING IN IMAGINATION)
  eval_return: 10 → 10 → 10 (FLAT IN REAL ENV)
  dynamics_loss: 0.3 (Low, seems good?)

Diagnosis: World model overfitted, policy exploiting model errors

       IMAGINATION (wrong)          REALITY (truth)

       Start: cart center          Start: cart center
          │                            │
     ┌────┴────┐                  ┌────┴────┐
     │ Policy: │                  │ Policy: │
     │Push left│                  │Push left│
     └────┬────┘                  └────┬────┘
          │                            │
          ▼                            ▼
    WM predicts: +10 reward       Real env: -5 reward
    (WRONG! Found model bug)      (Cart falls off!)

Solution: World model is making systematic errors
  1. Check if WM trained on diverse data (not just good trajectories)
  2. Increase WM capacity or train longer
  3. Add model regularization
  4. Use ensemble of dynamics models (vote on predictions)


FAILURE 3: Gradient Explosion
────────────────────────────────────────────────────────────────────
Symptoms:
  actor_grad_norm: 0.5 → 2.0 → 50.0 → NaN
  losses: All become NaN after ~1000 steps
  network outputs: NaN

Diagnosis: Gradients accumulating, exploding through imagination

Imagination rollout (15 steps):

  step 0: grad_norm = 0.5
  step 5: grad_norm = 5.0  (×10 growth)
  step 10: grad_norm = 500.0 (×100 growth)
  step 15: grad_norm = ∞ (EXPLODE)
          │
          ▼
    BACKPROP THROUGH ALL 15 STEPS
          │
          ▼
    Gradients multiply at each step
          │
          ▼
      EXPLOSION!

Solutions:
  1. Gradient clipping (already in code): max_norm=1.0
  2. Shorter imagination horizon: 15 → 10 or 5
  3. Lower learning rates: actor_lr 3e-5 → 1e-5
  4. Add gradient normalization


FAILURE 4: Critic Divergence
────────────────────────────────────────────────────────────────────
Symptoms:
  critic_loss: 1.0 → 5.0 → 20.0 → 100.0 (INCREASING)
  critic_value_mean: 10 → 100 → 1000 (EXPLODING)
  critic_td_error: 50 → 500 → 5000 (HUGE ERRORS)

Diagnosis: Critic overestimating values, entering feedback loop

  Iteration t:
    Critic predicts: V(s) = 100
    Compute target: R = r + γ·V(s') = 1 + 0.99·100 = 100
    Critic trained on: 100

  Iteration t+1:
    Critic predicts: V(s) = 105 (slight increase)
    Compute target: R = r + γ·V(s') = 1 + 0.99·105 = 105
    Critic trained on: 105

  Iteration t+2:
    Critic predicts: V(s) = 110 (keeps increasing!)
    ...
    RUNAWAY FEEDBACK LOOP

Solutions:
  1. Lower critic_lr: 3e-5 → 1e-5
  2. Use target network (update slowly)
  3. Clip critic targets: max=100
  4. Use λ-returns (already in code, good!)


FAILURE 5: Policy Convergence Too Fast
────────────────────────────────────────────────────────────────────
Symptoms:
  entropy: 1.5 → 0.5 → 0.01 → 0.0 (DROPS FAST)
  eval_return: 50 → 100 → 100 → 100 (PLATEAUS EARLY)
  policy outputs: Always same action

Diagnosis: Premature convergence, no exploration

  State: [any]  →  Policy: [0.001, 0.999]  →  Always action 1
                          (deterministic)

  Problem: Found local optimum, stopped exploring better solutions

Solutions:
  1. Increase entropy_coef: 0.01 → 0.05
  2. Lower actor learning rate (slower convergence)
  3. Use ε-greedy exploration (occasionally random actions)
  4. Adaptive entropy (SAC-style)
```

This completes the comprehensive gradient flow and debugging guide!
