Designing a Universal World Model Interface Based on SOTA Architectures

Universal World Model Architecture Design
Introduction

Building a universal world-model-based RL system means designing a modular architecture that can flexibly implement any SOTA world model algorithm (Dreamer, TD-MPC, IRIS, MuZero, etc.). The goal is to identify the common abstractions underlying these methods and expose configuration knobs for their key differences. This will allow rapid experimentation with new ideas by simply swapping components or toggling options, much like how your PPO implementation is structured modularly. The focus is on flexibility and ease of experimentation from day one, even if efficiency optimizations are refined later. In short, we want a single framework that can be configured to mimic Dreamer’s imagination-driven learning, MuZero’s planning with value-equivalent models, and everything in between, all through a coherent interface.

Flexibility and Experimentation as Top Priorities

Flexibility in architecture design directly enables faster experimentation. By decoupling components and designing clear interfaces, researchers can mix-and-match ideas without rewriting large portions of code. Since you haven’t yet hit performance bottlenecks with massive models, it’s wise to prioritize clean abstractions over micro-optimizations at this stage. A well-structured modular design will make it easier to incorporate future efficiency improvements when scaling up, without sacrificing clarity. In practice, this means accepting a bit of overhead or complexity now to ensure that as new world model ideas emerge (which they rapidly do), you can implement them by plugging into your framework rather than starting from scratch. The PPO subsystem in your codebase already follows this philosophy – it cleanly separates the encoder, policy, and value function, enabling different implementations to be inserted via config
GitHub
GitHub
. We want to achieve a similar level of interchangeability for world models.

Diverse World Model Approaches (SOTA Landscape)

To design a universal interface, we first survey the landscape of state-of-the-art world model algorithms to understand how they differ. Beyond Google’s Dreamer family, several fundamentally different approaches exist, each with unique design decisions:

Dreamer (PlaNet, Dreamer v1–v4) – These learn a recurrent state-space model (RSSM) with continuous latent states, reconstruct observations and predict rewards, and train an explicit policy via imagined rollouts (actor-critic in latent space). Dreamer v4 (2025) pushed this further by using a scalable transformer-based world model (replacing the RNN) with a shortcut forcing objective for fast inference
emergentmind.com
emergentmind.com
. Notably, Dreamer v4 can simulate complex 3D environments (Minecraft) entirely offline and still train an agent to achieve hard goals (like mining diamonds) using only imagination and offline video data
emergentmind.com
emergentmind.com
. Dreamer-style methods thus emphasize reconstruction and imagination: they learn to predict high-fidelity observations and rewards, and improve the policy by dreaming trajectories in the learned world model
emergentmind.com
.

TD-MPC Family (TD-MPC, TD-MPC2) – These methods use an implicit world model that forgoes observation reconstruction entirely. Instead of learning to decode images, the model learns a latent dynamics that is directly used for planning control signals. In other words, it’s a decoder-free world model focusing only on state transitions relevant to control
arxiv.org
. TD-MPC performs local trajectory optimization in latent space (essentially doing MPC on the learned dynamics rather than rolling out a policy)
arxiv.org
. The second iteration, TD-MPC2, scaled this approach to large models (317M parameters) and diverse tasks with a single set of hyperparameters
arxiv.org
, demonstrating the robustness of this design. Key distinctions of TD-MPC vs Dreamer:

No pixel reconstruction (the latent is implicit and not decoded to the original observation space)
arxiv.org
.

Planning via gradient-based optimization or shooting in latent space (rather than an learned policy generating actions step-by-step). In practice, TD-MPC’s controller optimizes a sequence of actions by backpropagating through the latent dynamics model, instead of using an imagination rollout with a parametric policy.

IRIS / $\Delta$-IRIS (Transformer World Models) – IRIS is a sample-efficient world model agent that uses a discrete latent representation of observations together with a Transformer dynamics model
arxiv.org
. Observations (e.g. Atari frames) are compressed via a VQ-VAE into discrete tokens, and an autoregressive Transformer predicts the sequence of tokens (dynamics) and rewards
arxiv.org
. This design casts world modeling as a sequence modeling problem, leveraging Transformers’ strength at long-range dependencies. IRIS achieved state-of-the-art human-normalized scores on Atari 100k with the equivalent of only 2 hours of real gameplay, outperforming prior methods in sample efficiency
arxiv.org
. The takeaway is that discrete latent spaces + Transformers are a viable alternative to RNN-based continuous latents. Any universal framework should accommodate tokenized observations and transformer dynamics, as used in IRIS. (Δ-IRIS is an improved version that likely builds on the same architecture with minor tweaks – meaning our design considerations for IRIS cover it as well.)

TransDreamer – This approach hybridizes Dreamer and Transformers. It keeps the overall Dreamer structure (learn model + train policy in imagination) but replaces the recurrent state model with a Transformer State-Space Model (TSSM) for the dynamics
arxiv.org
. TransDreamer also uses a transformer for the policy network, sharing the world model’s representations
arxiv.org
. The challenge addressed here was training stability and long-term memory: by using Transformers, TransDreamer can capture longer context than RNNs, and the paper shows improved performance on tasks requiring long-range memory compared to Dreamer
arxiv.org
arxiv.org
. This suggests our framework should allow swapping the sequence model (RNN vs Transformer) and still maintain integration with the policy/value learning.

MuZero (DeepMind) – A philosophically different approach, MuZero learns a value-equivalent model rather than an observation-equivalent model. It does not reconstruct observations at all; instead, the learned model focuses on predicting quantities relevant to planning: policy (action probabilities), value, and immediate reward
arxiv.org
. MuZero’s network has a representation function (encoder for the observation into a latent state), a dynamics function (predict next latent and reward given state+action), and a prediction function (output policy logits and value from latent)
arxiv.org
. During control, MuZero uses Monte Carlo Tree Search (MCTS) over the learned model: it expands possible action sequences in the latent space, using the predicted value to evaluate leaf nodes. This achieves superhuman performance in board games and Atari with no explicit environment model besides these predictions
arxiv.org
. The key design points for MuZero are:

No observation decoder and no reconstruction loss (the model is implicit about environment dynamics, only judged by how well it predicts rewards and values).

A planning Controller (MCTS) that queries the model iteratively rather than a single forward rollout.

The notion of a “world model” here is specialized: it’s trained via a value-aware loss (making the latent state good for value/policy predictions) rather than pixel prediction. Any universal framework must be able to skip reconstruction and plug in a planner in place of a learned policy network.

These examples illustrate the axes of variation among world-model algorithms. Dreamer and TransDreamer focus on learned imagination with full reconstructions, IRIS and TransDreamer bring in Transformers and discrete latents, TD-MPC foregoes decoding and uses analytic planning, and MuZero foregoes reconstruction and uses tree search planning. Despite differences, we can see abstract commonalities: they all have some form of encoder, some latent dynamics, some strategy to produce or select actions (policy or planner), and some training loop that may involve simulating trajectories (either by direct model rollout or by search). Leveraging these commonalities while allowing each method’s unique traits leads us to the next section.

Design Axes for a Universal World Model Framework

To accommodate the variety above, our framework should make the following design axes configurable via interfaces or parameters. These axes represent the major ways in which world model methods can differ:

Latent State Representation: What form does the internal state take? It could be a continuous vector (as in Dreamer’s RSSM) or discrete token sequence (IRIS’s VQ-VAE + Transformer) or even an implicit latent with no decoding (TD-MPC). We might need a LatentSpace abstraction that can handle different types (e.g. continuous tensors vs token sequences) and operations like batching, detaching, maybe sampling if stochastic. This ensures the agent’s core can operate on either type of latent. For example, IRIS’s use of a discrete autoencoder means the framework must accept latents that are indices of codebook entries
arxiv.org
, while Dreamer’s can be a vector from an RSSM. The LatentState concept can wrap these and offer a consistent API (like .sample() if probabilistic, or conversions to/from raw features).

Dynamics / Transition Model: How do we predict the next state? Different algorithms use different transition models:

Recurrent state-space model (RSSM) with stochastic and deterministic components (Dreamer V2/V3 style).

Transformer dynamics (IRIS, DreamerV4’s block-causal transformer
emergentmind.com
, TransDreamer’s TSSM
arxiv.org
).

Implicit or feedforward latent dynamics (TD-MPC uses a deterministic latent transition without an explicit output distribution for pixels
arxiv.org
, often training it with consistency or value losses).

MuZero-style value-equivalent dynamics, where the model predicts next latent and reward but is not required to mimic pixel dynamics.
We should define a DynamicsCore interface that exposes at least a step(prev_state, action) -> next_state (and maybe reward dist) method. This could be implemented by an RNN-based model, a Transformer, or any custom latent transition. It might also include methods like rollout (for models that can simulate multiple steps internally) or a flag for shortcut/skip connections (Dreamer4’s shortcut forcing objective effectively trains multi-step predictions by combining shorter predictions
emergentmind.com
 – the interface could allow injecting such training logic if needed, e.g. a special loss or multi-step prediction mode). By abstracting the transition, we can plug in an RSSM for one experiment and a Transformer for another, without changing the surrounding training code.

Observation and Reward Prediction: Does the model reconstruct observations and predict rewards, or not? In a Dreamer-style model, observation reconstruction and reward prediction are part of the loss (world model is trained to predict pixels and rewards over the imagined trajectory). In MuZero, no observation is predicted, only immediate reward is predicted inside the model
arxiv.org
. TD-MPC similarly doesn’t reconstruct the observation space at all (no decoder)
arxiv.org
. Our framework should allow optional decoding heads for observations and rewards. We can have a set of Predictive Heads (e.g. ObservationDecoder, RewardModel) that can be attached or omitted. These would take the latent state and output a reconstruction or reward estimate. The training loss computation can automatically include the reconstruction loss if a decoder is present (as your current code does via representation_learner.decode() if available
GitHub
). If no decoder is present (e.g. in a value-equivalent setup), that loss term is simply absent. By modularizing this, we ensure algorithms like MuZero aren’t forced to waste capacity on decoding pixels, while algorithms like Dreamer can still train the latent by reconstruction loss. Similarly, a RewardHead could be included for models that learn a separate reward predictor (some Dreamer versions and MuZero do train a reward model explicitly), but omitted if the dynamics directly outputs reward or if using ground-truth rewards for model loss.

Control/Policy Strategy: How are actions decided and which component is responsible? We have two broad classes:

Policy Networks (Actors): Dreamer uses a learned policy network (often called the actor) that takes the current latent state and outputs an action distribution. IRIS also ultimately trains a policy in latent space (after using imagined trajectories). In these cases, the agent’s decision-making is a neural network mapping state -> action, learned via gradient descent (reinforce, actor-critic, etc.).

Planners (MPC or Tree Search Controllers): Other methods choose actions by planning on the fly. TD-MPC uses an MPC-like procedure: at each time step, it optimizes a sequence of actions by interacting with the learned dynamics (often via backprop through the model, sometimes called implicit differentiation planning)
arxiv.org
. MuZero uses MCTS over the learned model
arxiv.org
. In our architecture, we should introduce a Controller abstraction that covers both cases. A Controller.act(latent_state) method could either:

Internally query a learned policy head (for actor-based methods), or

Run a planning algorithm using the dynamics model and value function.
We may implement a BaseController with subclasses like PolicyController (just wraps a policy network) and PlannerController (for MPC or MCTS). In your current code, the WorldModelParadigm.forward() already chooses between a planner vs direct policy if a planner is provided
GitHub
. We’d extend this idea: the paradigm can always call action = controller.act(state), and under the hood the controller either does policy_head(state) or something more elaborate. This clean separation means an experimenter can switch from learning a reactive policy to doing on-the-fly planning by just swapping the controller type in config.

Training Regime / Loss Computation: Different algorithms optimize different loss terms and possibly in different sequences:

Dreamer (v2/v3) trains world model and policy/value together online (interleaving environment collection with imagination updates). Dreamer v4 does two-stage training: first train the world model on offline data, then freeze or slow it and train a policy head on imagined data
emergentmind.com
emergentmind.com
.

MuZero alternates between self-play (to generate data) and training steps that use a combination of MCTS target value and policy losses.

Some methods incorporate additional losses (e.g., consistency regularizers, or inverse model losses for representation learning, etc.).
A universal framework might need a flexible way to orchestrate training phases and loss terms. One idea is a TrainingGraph or scheduler object that defines which losses to compute and when. However, this can also be handled by configuration: e.g., a YAML could specify that world model training is warmed up for N epochs before enabling policy learning, etc. At minimum, the WorldModelParadigm.compute_loss() should be refactored to not be one monolithic Dreamer-specific computation
GitHub
. Instead, it could call into sub-modules: e.g., losses = dynamics_core.loss(...) + decoder.loss(...)+ value_head.loss(...)+ policy_head.loss(...)+ planner.loss(...) depending on what’s present. Each component can contribute its terms. This decoupling will let us accommodate new loss terms without rewriting the whole function. For example, if we implement a MuZero variant, we might skip reconstruction loss and use a policy/value loss based on MCTS results instead of imagined trajectories – the framework should allow that swap cleanly. Declaring training phases (like “world model pretrain phase” vs “policy fine-tune phase”) could also be useful; this could be done with simple flags in config or by a more complex training loop manager.

Rollout and Imagination Style: In Dreamer, the rollout in imagination is core – you simulate many steps into the future with the learned model and use those to train the policy (and value)
GitHub
. In contrast, a planner like TD-MPC or MuZero doesn’t roll out entire trajectories for training the policy in the same way; instead, TD-MPC does short optimizations at each step, and MuZero uses a tree search to evaluate actions. Our framework should treat “imagining trajectories” as one possible mode. Perhaps the Controller or DynamicsCore can have a method to generate trajectories. Another approach is to have a Simulator utility that given the dynamics (and maybe a policy) can produce a trajectory in latent space. Dreamer’s use-case: simulate N steps with the current policy (this we have in rollout_imagination currently
GitHub
GitHub
). Planner use-case: we might not need long rollouts, but we might still want to simulate a few steps to evaluate action sequences. We should design the interface such that the length and use of imagined rollouts is configurable. For example, a config for Dreamer could say imagination_length: 15 and use policy rollout, whereas a config for TD-MPC might say imagination_length: 1 (or none) and instead rely on the controller’s optimization. In practice, we could incorporate imagination into the Controller for planner-based methods – e.g., a PlannerController might internally simulate trajectories of a given horizon to evaluate them (so it still uses the model, but in a different way than an actor). Ensuring that our WorldModelSystem can either roll out with the policy or allow the planner to query the model is essential.

In summary, these axes (latent representation, transition model type, prediction heads, control method, training regime, and rollout style) cover the spectrum of design decisions found in current world model methods. By exposing each axis as a pluggable module or option, we "span the space" of possible architectures. This prevents the framework from being overly tied to one paradigm (like the current code is tied to Dreamer-like imagination and reconstruction
GitHub
GitHub
) and makes it future-proof as new algorithms often combine elements across these axes.

Core Abstractions and Modular Components

To implement the above flexibly, we will introduce clear abstract components in the code. Each component corresponds to one axis or functional part of the system. Here are the core abstractions we should define (likely as base classes or interfaces), along with their roles:

LatentState & LatentSpace: A pair of classes (or a combined one) to represent the latent state. This could be a simple wrapper around a torch.Tensor for continuous states, or something more complex for discrete sequences (e.g., containing token indices plus perhaps the encoder’s codebook for decoding if needed). The idea is to have methods that all latent states should support, such as .detach(), .to(device), or concatenation, etc., abstracting whether the latent is a vector or sequence. A LatentSpace class might handle the creation of latent states or define dimensions. This abstraction helps especially if we need to project latents to different forms (for example, MuZero might want a latent for MCTS that excludes certain parts of state, or TD-MPC might need to detach gradients for planning). By formalizing latent states, we avoid hardcoding torch.Tensor assumptions everywhere and make it easier to extend to new latent types.

ObservationAdapter (Encoder & Preprocessing): This component will encapsulate how raw observations are turned into the initial latent state. In your current code, this is split between an encoder (image encoder or feature extractor) and a representation_learner which might do further processing or reconstruction
GitHub
. We can combine these into an ObservationAdapter pipeline: it takes an observation (maybe a dict of modalities or just an image tensor) and outputs the latent state (or latent features before the dynamics). Under the hood, it could apply augmentations, encode the observation, and if using a two-step representation (like RSSM uses an encoder then a deterministic/stochastic split in the RSSM), it would manage that. Essentially, this ensures no matter what front-end processing an algorithm needs (VQ-VAE for IRIS, ResNet encoder for Dreamer, etc.), the rest of the system just deals with the resulting latent. We might implement this by having the encoder and representation_learner work in tandem. For example, in Dreamer, representation_learner.encode(features) produces the initial state RSSM. In IRIS, the encoder might do VQ tokenization, and representation_learner could be identity (or just the VQ itself). An ObservationAdapter class could formalize this pairing and produce a unified EncoderOutput object (containing latent state and maybe any extra info like tokenization masks). This is mostly an organizational improvement to simplify how we call encoding in the paradigm.

DynamicsCore: This is the heart of the world model – an interface that all dynamics models (RSSM, transformer models, implicit models, MuZero models) will implement. At minimum it should provide step(latent_state, action) -> next_latent (and optional outputs) method. It might also provide:

init_state(...) if an algorithm needs to initialize a latent from an observation (though that could also be part of the representation learner).

loss(predicted_next, actual_next) or a more general dynamics_loss(states, actions, next_states) – your code currently calls dynamics_model.dynamics_loss(...) to get the model’s loss
GitHub
. Each implementation can define its training loss: e.g., an RSSM might include KL divergence between posterior and prior; a transformer might use cross-entropy on token prediction; MuZero might not use a traditional dynamics loss but could use value consistency losses.

If supporting advanced features like Dreamer4’s shortcut forcing, the DynamicsCore could have a flag or method to do multi-step predictions for training. Alternatively, that can be handled inside its dynamics_loss implementation (the Dreamer4 model could compute its multi-step loss internally given a sequence of states).
We may also want a method to rollout a trajectory for imagination if the dynamics model can be unrolled independently (though typically we roll it out by repeatedly calling step with a policy). Perhaps more useful is ensuring DynamicsCore can interact with planners: e.g., we might need a function to get gradient information through the model for planning (TD-MPC needs gradients of value w.rt. action via the model). We can expose a method for that or simply rely on PyTorch autograd with the model’s step being differentiable. In any case, DynamicsCore is central – it lets us plug in anything from a simple deterministic function to a huge transformer, as long as the interface is obeyed.

Predictive Heads (Decoders and Heads): To handle observation reconstruction, reward prediction, value estimation, policy logits, etc., we create a set of head modules. You already have BasePolicyHead and BaseValueFunction classes in components, which are used in PPO and in WorldModelParadigm
GitHub
. We should likely extend this idea to world-model-specific heads:

ObservationDecoder: produces reconstructed observation (for pixel-based losses).

RewardModel: predicts immediate reward (if using learned reward).

ValueFunction (Critic): estimates cumulative value from state (already have this).

PolicyHead: outputs an action distribution from state (already have this).

Possibly ConsistencyHead or InverseDynamics: if we consider self-supervised losses like contrastive state consistency or inverse model losses (some frameworks add these for representation learning). Codex mentioned a ConsistencyHead – this could be a placeholder for any extra loss that enforces consistency between imagined states and real ones (for example, a loss used in some algorithms to align latent with reality without reconstruction).
Each head should implement a common interface, perhaps a forward(latent) -> prediction and a loss(prediction, target) method. We can register these heads in a registry and include them as needed. In practice, Dreamer’s config would include an ObservationDecoder and RewardModel; MuZero’s config would exclude those (or use none) and include only Policy and Value heads (since MuZero’s world model predicts reward internally, we might treat that as part of dynamics or a separate small head). By structuring heads modularly, adding a new output (say some future algorithm wants to predict “discount factor” or some auxiliary value) becomes easy.

Controller (Policy or Planner): The Controller unifies the action selection process, whether it’s a learned policy or a planner. We can define a BaseController with an interface like act(latent_state, deterministic=False, horizon=None, **kwargs) – where horizon might be relevant for planners (e.g., how far to look ahead or how many steps to optimize). For a policy controller, this act just calls the policy head’s distribution and samples an action (or argmax if deterministic). For an MPC controller, act would run the optimization routine: simulate many candidate action sequences using the dynamics model, evaluate them (perhaps using the value function at the end of the horizon), and return the first action of the best sequence. For an MCTS controller, act would run MCTS with the dynamics and value/policy heads guiding it. The rest of the system doesn’t need to know which it is – the WorldModelParadigm can simply use action = controller.act(state) in its forward pass. In your current code, this logic is partially there: if self.planner exists, it calls self.planner.plan(states, dynamics_model, value_function)
GitHub
. We would refine that by integrating the planner into the Controller interface (and possibly rename planner to something like MPCPlanner implementing BaseController). This way, the policy training loop can also be abstract: for an actor-based method, training the policy happens via backprop through the policy network (as done in Dreamer/PPO losses), whereas for a planner-based method, “training” might simply not exist (you don’t train the planner, but you might train the world model or value model from the planner’s outcomes, such as MuZero training its value network from MCTS results). The framework could accommodate this because policy loss computation would be different depending on controller type – e.g., if controller is a Planner, the “policy_loss” term in Dreamer sense might be irrelevant, but we might have other losses (like consistency or supervised expert imitation loss, etc. depending on algorithm). We ensure this by designing the training loop to ask the controller or config what losses to include. For example, we could have the planner contribute a loss (MuZero’s planner doesn’t have a loss, but MuZero’s value network does, which we already handle via value_loss). We might treat “planner” differently by not treating it as a learnable component (just as an acting component). So the Controller mainly affects how actions are produced during rollouts and deployment, and what kind of loss is used for the policy.

Training Phase Manager / Graph: To handle various training regimes, we can introduce an abstraction to coordinate training phases and loss combinations. One approach is to create a TrainingGraph or LearningScenario object that the paradigm uses to compute losses. For example, in a Dreamer config, the TrainingGraph might specify: world_model_loss = reconstruction_loss + dynamics_loss + reward_loss; actor_loss = - E[logπ * λ * advantage] (from imagination); value_loss = MSE(predicted value, imagined returns). In an offline learning config like Dreamer4, the TrainingGraph might specify a pretraining phase where only world_model_loss (reconstruction + dynamics) is optimized, and a fine-tuning phase where those are fixed and only policy/value losses are optimized
emergentmind.com
emergentmind.com
. This could be implemented as simple flags (e.g., an offline_pretrain_steps setting after which we start including policy loss), or as a more declarative list of phases with their own optimizers. Codex suggested using the existing config system to list which losses and phases to run, possibly integrating with your YAML experiment definitions. Given your registry system, we could allow something like:

paradigm_config:
    phases:
      - name: "world_model_pretrain"
        updates: ["dynamics", "encoder"]   # components to train
        losses: ["reconstruction_loss", "dynamics_loss"]
        duration: 10000  # steps
      - name: "policy_training"
        updates: ["policy_head", "value_function"]
        losses: ["policy_loss", "value_loss", "reward_loss"]


This is an advanced feature; initially, we might hard-code simpler logic (like a boolean offline_stage flag). The key point is our architecture should separate the concept of world model update and policy update, so we can support both one-stage (online Dreamer) and two-stage (pretrain then finetune) schemes. Internally, the WorldModelParadigm can hold a WorldModelSystem (see below) that can be trained in different modes as configured.

WorldModelSystem Orchestrator: Finally, we can create a high-level orchestration class (Codex called it WorldModelSystem) that wires together all the above pieces. In effect, this might be what WorldModelParadigm becomes (or owns). It would take instances of the encoder/adapter, dynamics core, heads, controller, etc., and provide top-level methods like predict_action(obs) for acting and compute_losses(batch) for training. This system ensures all components are aware of each other if needed (for example, the planner might need to query the dynamics and value function – we can pass those references in). This class could also implement utility methods like simulate_trajectory(initial_obs, policy, horizon) which could be useful for evaluation or debugging. By grouping everything in a system class, WorldModelParadigm (which is part of your BaseParadigm hierarchy) can delegate to it. This keeps the paradigm class itself fairly thin and focused on integration with the rest of your RL loop (e.g., interacting with buffers, calling optimize). In other words, WorldModelParadigm becomes a container that instantiates the configured WorldModelSystem via factories and uses it. This approach will make it easier to extend or even create new paradigms. For instance, if someday you want a hybrid model-free + model-based paradigm, you could instantiate both a model-free system and a world-model system and coordinate them – having them encapsulated helps manage complexity.

All these abstractions aim to make the architecture highly modular. Each corresponds to one dimension where algorithms vary, and by coding them separately, we allow combinations. To illustrate the power: one could configure a system with a discrete latent (IRIS) + transformer dynamics + MCTS controller – even if no published paper exactly did that, the framework would support experimenting with it. This modular design is analogous to how in PPO you separate policy and value networks; here we just have more pieces.

Mapping SOTA Models to the Abstractions

It’s useful to verify that our proposed components can indeed recreate the major algorithms in a clean way. Here’s how each of the earlier-mentioned methods would map into our modular design (as a sanity check):

Dreamer (v1–v3):

LatentSpace: Continuous vector (RSSM state with mean and std for stochastic part).

DynamicsCore: RSSM recurrent model (with both deterministic RNN state and stochastic posterior/prior). It would implement step(state, action) using the RSSM transition (and would likely require the previous posterior sample, etc., but that’s internal details). It would also provide a dynamics_loss combining reconstruction error of observations, reward prediction error, and a KL regularization term between posterior and prior (to keep the model realistic).

ObservationAdapter: CNN or MLP encoder for observations; representation_learner that outputs the initial RSSM state (Dreamer’s “representation network”).

PredictiveHeads: ObservationDecoder (to reconstruct pixels), RewardHead (to predict immediate reward from latent), ValueHead (to estimate state value), PolicyHead (actor network for action distribution).

Controller: a PolicyController that just queries the PolicyHead (since Dreamer’s action comes from the learned actor). No external planner.

Training: online actor-critic style. The loss terms would include reconstruction loss, reward prediction loss, value loss (e.g. dreamer critic loss), and policy loss (dreamer actor loss which maximizes expected value and includes an entropy term) – all of which our framework can sum up. Imagination rollout is used to compute advantages or returns for the actor (e.g., Dreamer uses $\lambda$-returns in latent imagination). We would simulate trajectories via DynamicsCore and use PolicyHead to sample actions (this is essentially what rollout_imagination already does in your code
GitHub
GitHub
).

Dreamer v4: This is essentially Dreamer with two notable changes: (1) Transformer dynamics in place of RSSM, and (2) a distinct two-phase training (offline pretrain on video, then RL fine-tune in latent). In our framework:

We’d substitute the DynamicsCore with a TransformerDynamics class. It might treat a sequence of past observations (or latent tokens) as context and predict next token or frame. It would likely operate in large temporal batches rather than step-by-step RNN style. The block-causal transformer with shortcut forcing can be integrated by having its dynamics_loss implement the shortcut forcing objective (predict low-level latent with partial sequence, etc.)
emergentmind.com
. But outside modules (controller, heads) remain similar.

The ObservationAdapter in Dreamer4 includes a Causal Tokenizer (like a VQ-VAE + masking for high-res frames)
emergentmind.com
. We could implement that as part of the encoder stage.

Predictive heads: Dreamer4 also uses learned reward and policy heads after pretraining
emergentmind.com
 – those we have.

Controller: still a learned policy (no external planner).

TrainingGraph: we’d configure an offline pretraining where we optimize observation and reward prediction (world model training) using logged data, then enable the imagination RL phase. Our framework’s ability to toggle losses/phases handles this (e.g., do not apply policy loss until phase 2). This demonstrates why a phase-manager is valuable.

TD-MPC / TD-MPC2:

LatentSpace: Likely a continuous vector (embedding of observation) but importantly no decoder. The latent is implicit.

DynamicsCore: An implicit latent dynamics model. Possibly a deterministic function $f(s,a)\to s'$ (no stochastic sampling, since they avoid reconstructions). It might output a next latent state directly. The dynamics_loss here might be a temporal difference loss: TD-MPC trains the model such that a value or Q estimate at $s'$ matches a target (coming from TD learning) – effectively blending model learning with value learning. They also likely enforce some consistency (e.g., the Jacobian of dynamics to support gradient-based planning). We might implement this dynamics model class to return not just next state but also allow computing gradients of a value head through it. This is advanced but doable: our interface can leave it to the user to configure a differentiable planner.

PredictiveHeads: No ObservationDecoder (we skip it). Possibly no RewardHead either if rewards are given by environment (or we could have a trivial one that just passes through if we want uniform interface). They do have a value function (for evaluating sequences) and sometimes a policy network for initial action guess, but core is planning, so probably we consider only a value head.

Controller: a PlannerController implementing the latent trajectory optimization. At each act(), it could simulate say $H$ steps into the future in latent space (using the dynamics model) with many random or learned candidate action sequences, evaluate their cumulative reward or value at the end (using the value head for long-term), and pick the best first action. This can be done by gradient descent on actions or just sampling (the TD-MPC paper uses gradient and cross-entropy methods). Our planner can be generic enough to allow either.

Training: The loss would consist of something like: a TD error for the value (or Q) function, plus perhaps an imitation of the planner’s choice by the policy head (TD-MPC2 might train a policy network to mimic the planner for efficiency, though not sure). The framework would need to support a value loss that treats the dynamics model as part of the computation (which it already can, since we have value_function and dynamics_model – we just need to feed appropriate targets). We might also incorporate a consistency loss on latent (ensuring the latent doesn’t drift too much). In any case, the absence of reconstruction is naturally handled by not including that head/loss. The planning integration is the main addition, which our Controller abstraction covers.

IRIS:

LatentSpace: Discrete tokens (from a VQ-VAE). We’d have an Encoder that encodes an image to, say, a $32\times32$ grid of codes (or whatever their spatial downsampling is) and yields a sequence of tokens. The LatentState could hold these token IDs and maybe also their embedding vectors if needed.

DynamicsCore: An autoregressive Transformer that takes the sequence of past tokens (and possibly recent actions) and predicts the next token (or next sequence of tokens). Essentially it’s a sequence model that unrolls one step at a time, predicting the next observation’s latent code distribution and reward. IRIS being an offline method, its training of dynamics is supervised from real trajectories (similar to a language model predicting next token given previous). So dynamics_loss here could be cross-entropy of predicting the actual next token (from the VQ code of the next observation)
arxiv.org
. They achieved high accuracy with limited data, indicating this approach is strong.

PredictiveHeads: IRIS likely does not decode tokens back to pixel images during training except perhaps to visualize; the loss is on token prediction rather than pixel MSE. But one could include an ObservationDecoder to map tokens to pixels if needed (to measure image-space loss too). We can decide to either treat the token predictor as the dynamics model itself (i.e., the model outputs token logits and we apply cross-entropy), or have a head that takes latent and outputs reconstruction (which in this case would be one that maps the token latent to pixel – basically the VQ-VAE decoder). IRIS’s reward prediction could be a small head on the latent as well. Value head might not have been used in IRIS because it was focusing on short-horizon tasks (Atari 100k, typically they still did planning or policy?). Actually, IRIS did train a policy on imagined trajectories (they mention “trained over millions of imagined trajectories” in some summaries). So yes, after training the model, they did an imagination-based policy training (like Dreamer) but with the transformer model generating trajectories.

Controller: Likely a learned policy (actor) because IRIS did not do search; it was model predictive but via policy gradients in imagination (and they note “without lookahead search” in achieving state of art
arxiv.org
). So we use a PolicyController.

Training: Phase 1: train world model (VQ-VAE + Transformer) on real data (minimizing reconstruction/token prediction loss). Phase 2: fix world model, then do imagination rollouts in latent token space to train the policy and value via RL (basically Dreamer but with a transformer model). Our architecture handles this by configuring a transformer DynamicsCore and discrete latent, plus using the same policy/value training loop as Dreamer. The key difference is we must ensure our imagination rollout works with discrete latents (which is fine if the dynamics core returns token distributions and we sample tokens to get next state). We should also support the fact that the transformer might need a history window – TransDreamer addressed this by sharing policy and model to avoid divergence
arxiv.org
. We might need to feed the transformer model the sequence each time. This can be handled either internally (dynamics keeps its internal state like an RNN, or we always feed full context of recent N steps). These are details that our design can accommodate with careful API (maybe the LatentState carries a sequence context for transformer models).

MuZero:

LatentSpace: Usually a continuous vector (often small, like hidden state of a ResNet). We set it as a deterministic latent (no stochastic component).

DynamicsCore: A MuZero model would implement step(s,a) producing s_next and also an immediate reward prediction. We could model the reward prediction as part of DynamicsCore’s output or as a separate RewardHead that takes $(s,a)$ or $s'$ and outputs reward. In MuZero’s original network, the dynamics function outputs the next latent and reward in one go
arxiv.org
, so we might just integrate reward into the dynamics model’s output for fidelity. The dynamics_loss would not try to reconstruct observations (no decoder), but would include the error in reward prediction (difference between predicted reward and actual environment reward for the transition) and perhaps a consistency penalty to ensure the latent can predict the next state’s value accurately (MuZero is trained end-to-end by aligning predicted value with actual returns, effectively).

PredictiveHeads: We would include only the ValueHead and PolicyHead. The policy head gives logits for actions from a state (this corresponds to the “prediction network” in MuZero that given latent state outputs policy and value). The value head gives the state value. No observation decoder, no separate reward head if we folded reward into dynamics; if not, we’d include a RewardHead as well. But note, MuZero’s reward prediction is one step, which is low-dimensional, so including it as part of dynamics or as a tiny head is equivalent in design terms.

Controller: A PlannerController implementing MCTS. During act(state) it will perform the tree search: simulate many possible action sequences using the DynamicsCore repeatedly. It will use the PolicyHead & ValueHead to guide and evaluate the search (just like MuZero does – the policy head’s prior over actions at a node guides which actions to explore, the value head estimates value at leaf nodes)
arxiv.org
. After search, the controller returns the selected action (e.g., the root action with highest visit count).

Training: MuZero training involves using the MCTS returns as targets for the network. So for each real game step, they have a target value (the n-step or Monte Carlo return), target policy (the MCTS visit distribution), and target reward (the actual next reward). The losses are: value loss (predicted value vs target value), policy loss (predicted policy vs MCTS policy, e.g., cross-entropy), reward loss (predicted vs actual reward). These are all handled by our heads: ValueHead.value_loss can cover value target
GitHub
, PolicyHead could have a loss comparing to target policy (though in our current setup policy_head is used for sampling; we might need to extend it to supervise it with a target distribution – this can be done by a custom loss if needed). Our framework can accommodate this by a custom compute_loss for MuZero paradigm or simply by configuring the loss differently in TrainingGraph (since MuZero doesn’t use imagination rollouts at all for training, we wouldn’t call rollout_imagination in that configuration). Instead, training is supervised from real transitions augmented by MCTS – that might require a slight integration to store the MCTS result in the batch (e.g., batch contains target_policy and target_value). We can handle that by allowing compute_loss to check if those exist and then do appropriate loss calculations. So the modular design doesn’t break here – it just means in MuZero’s case, the world model system’s training loop is different. We might implement MuZero as a separate subclass of WorldModelParadigm that overrides compute_loss; or better, parameterize the training loop via config. Either way, the core pieces (dynamics with no decoder, heads for value/policy, planner controller) are in place.

Through these mappings, we see our abstractions are sufficient to express each algorithm’s requirements. It also highlights that our framework would make it easy to mix elements (for example, one could try MuZero’s value-equivalent approach but with a learned policy head instead of MCTS, or Dreamer’s imagination but with a discrete latent). This is a strong argument for universality – research is often about combining ideas from different sources, and a unified framework accelerates that.

Design Considerations: Balancing Generality and Simplicity

Designing such a universal system brings both advantages and challenges. It’s important to acknowledge these to ensure we make informed decisions:

 

Advantages of a Modular Unified Design:

Rapid Innovation: New ideas can be implemented by swapping out a module or adding a new one, without rewriting the entire algorithm loop. For instance, if a “Dreamer 5” comes out requiring a new loss head or a different planner, you can integrate it by implementing the specific component and plugging it into the existing interface, rather than creating a whole new training pipeline. This greatly accelerates research iterations
GitHub
GitHub
.

Code Reuse: Common functionality (like encoding observations, or calculating value losses) is written once and reused across algorithms. This avoids duplicate code for each new agent, reducing bugs and technical debt. It also means improvements (like a better optimizer schedule or a bugfix in a loss calculation) propagate to all agents that use that component.

Consistent Interface: Experimenters and collaborators only need to learn one API. If someone knows how to configure and run a Dreamer agent in your framework, they can use the same knowledge to set up a MuZero agent by just changing the config. This lowers the barrier to entry for experimenting with different approaches.

Mix-and-Match Experiments: The biggest research payoff is the ability to combine elements in novel ways. For example, one could test MuZero’s MCTS planner on top of a Dreamer learned model, or use a VQ-VAE discrete latent in the Dreamer imagination framework. With a modular design, such hybrid experiments are trivial config changes, which could lead to new insights or performance gains.

Maintainability: Clear separation of concerns (encoder vs dynamics vs controller, etc.) makes the codebase easier to maintain. If a bug is in the planning logic, it will likely reside in the PlannerController, isolated from the dynamics code. This modularity makes debugging easier. It also means one can update one part (say, improve the encoder architecture) and as long as it adheres to the interface, everything else continues to work.

Challenges and Potential Criticisms:

Complexity and Overhead: The abstraction layers introduce complexity. More classes and interfaces can be harder to understand initially than a straightforward monolithic implementation of one algorithm. There’s a risk of “over-engineering” – designing for generality that isn’t needed if you only ever use a subset of features. We should be mindful to keep defaults sensible so that the common case (likely Dreamer-like use) isn’t cumbersome. The code should still be approachable for someone who just wants to run a standard algorithm.

Performance Considerations: A universal framework might incur some runtime overhead or constrain certain optimizations. For example, if our dynamics interface is too generic, we might not fully exploit parallelism or sequence batching that a specialized implementation would. However, these costs can often be mitigated with careful engineering (e.g., the Dreamer4 transformer can still be used efficiently as long as we allow passing sequences to it). Since we are not yet at massive scale, this is a secondary concern, but we should design with an eye towards not painting ourselves into a corner. We should ensure that critical inner loops (like imagination rollout or MCTS planning) can bypass Python-level loops when needed (perhaps by vectorizing or C++ bindings in the future).

Edge Cases and One-off Logic: Some algorithms have unique tricks that don’t map cleanly to a generic interface. For instance, MuZero’s approach to training involves storing search statistics as targets – our framework might need some custom handling for that (like passing extra data through the batch). We might end up needing a few if/else in the training loop depending on config flags (e.g., if use_mcts: do X). We should try to minimize these, but some conditional logic is acceptable to keep the system flexible. The key is to avoid hardcoding a specific algorithm’s hack in a way that affects others. As another example, Dreamer v2/v3 had the concept of exploration noise in policy or KL balancing in the loss – we should integrate such options cleanly (perhaps as parameters to loss functions).

Learning Curve for Contributors: New developers or researchers might need to understand the abstractions to use the framework properly. Good documentation and examples will be important. We should provide clear “cookbook recipes” for how to set up each known architecture (Dreamer, MuZero, etc.) so users can start from those and only tweak what they need. This mitigates confusion and shows that despite the generality, it’s straightforward to instantiate a specific instance.

Modularizing vs. Custom Optimization: Sometimes an algorithm achieves its performance through tightly coupled logic between components. When we modularize, we might inadvertently restrict that. For instance, Dreamer’s imagination rollout and actor update are somewhat intertwined – by abstracting them, we must ensure the result is the same. Another example is training stability tricks: TransDreamer had to share weights between model and policy to stabilize transformer training
arxiv.org
 – our design would normally separate dynamics and policy, but we should allow an option to have them share parameters (maybe via giving the same backbone to both). This is an advanced use-case, but it’s worth noting so we don’t assume policy and model are always independent. We can handle it by letting the policy head optionally be the same object as part of dynamics (config could specify that). It’s a corner case but shows the need for some flexibility beyond strict module boundaries.

Overall, the benefits of a well-abstracted system likely outweigh the downsides, especially for a research codebase where flexibility = velocity. By anticipating the challenges (as above), we can design mitigations (good defaults, optimize critical paths, allow escape hatches for special cases). The result should be a framework that is both powerful and usable.

Implementation Plan

To achieve this step-by-step, we can outline a development plan. Breaking it down will help ensure we maintain a working codebase throughout the refactor:

Define Core Interfaces and Classes: Start by creating the skeleton classes for the new components in a new module (e.g., src/world_models/). For example:

LatentState (with subclasses ContinuousLatent, DiscreteLatent if needed).

BaseDynamicsModel (could reuse the existing BaseDynamicsModel but extend it for new interface).

BaseController and perhaps PlannerController and PolicyController subclasses.

ObservationDecoder, RewardModel classes (subclasses of a common base like BasePredictiveHead).

Possibly a WorldModelSystem class that has attributes for all parts (encoder, dynamics, etc.) and a method like compute_loss(batch) that uses those parts.
Define these mostly as placeholders with method signatures and documentation. This creates the “shape” of the system without full functionality, and we can register them in the registry (so config can refer to them).

Refactor WorldModelParadigm to use the new classes: Gradually migrate the existing WorldModelParadigm implementation to delegate to the new abstractions. For example, currently WorldModelParadigm.compute_loss directly computes everything
GitHub
. We can instead implement a new WorldModelSystem.compute_loss that does similar, but by calling sub-component methods. Initially, implement it to achieve parity with the current behavior (Dreamer-style) to ensure we don’t break existing training. That means:

Make the encoder + representation produce a state.

If a decoder head is present, compute recon loss.

Use dynamics_model.dynamics_loss for model loss.

Value loss if returns given.

Policy loss via imagination if using a policy controller (like now).
We should test that after refactor, using a config for Dreamer (or the placeholder world model you have) yields the same losses as before on a sample batch. It might be wise to include a temporary flag like legacy_mode=True to verify we haven’t changed outcomes. Also, ensure that if no decoder is present, the code skips that loss gracefully (it already does with the hasattr(decode) check
GitHub
, which we can replace with a more general conditional based on heads configured).

Introduce Configuration for New Axes: Extend the YAML config schema and the factory to handle the new components. For instance, in your ComponentFactory.create_paradigm, currently it looks for dynamics_model, value_function, etc.
GitHub
. We’ll add entries for any new components like observation_decoder or specialized controllers. We might nest these under the paradigm config. For example, the config could look like:

paradigm: "world_model"
encoder: {type: "resnet", config: {...}}
representation_learner: {type: "rssm", config: {...}} 
dynamics_model: {type: "rssm_dynamics", config: {...}}
policy_head: {type: "gaussian_mlp", config: {...}}
value_function: {type: "critic_mlp", config: {...}}
# New heads could be optional:
observation_decoder: {type: "conv_decoder", config: {...}}
reward_model: {type: "dense_reward", config: {...}}
controller: {type: "policy", config: {...}}  # or "planner", etc.
paradigm_config: { imagination_horizon: 15, ... other high-level settings ...}


Update the factory to create these if present and pass them into WorldModelSystem (or WorldModelParadigm). This step is mechanical but important for usability: all new components must be easily specifiable in configs.

Implement Two Reference Configurations & Test: As a proof of concept, implement two different world model configurations to ensure the framework covers them:

A Dreamer-like config (could be a “toy Dreamer” on a small environment). Use continuous latent, an RSSM or simple RNN dynamics, include decoder & reward head, and a policy controller. Train it on a simple task (like CartPole or a tiny gridworld) to verify that the imagination rollout and learning work as expected. Compare its performance or at least its loss curves to an expectation (or to the pre-refactor version) to ensure nothing fundamental broke.

A TD-MPC-like or MuZero-like config (a simple version). For example, configure a latent without decoder, use a planner controller. Possibly test on a very simple domain where planning is obviously different from a reactive policy (like a maze or something). This is more challenging to test without a lot of additional code (like implementing MCTS). Alternatively, implement a trivial planner that, say, looks one step ahead using the model’s reward prediction (just to test the plumbing). The goal is to make sure the code can handle the absence of certain components (no decoder, etc.) and the presence of a planner.
If doing MuZero-lite: we could do something like a small deterministic environment (e.g., predict a known sequence) and see if the planner selects correct actions. These tests can be unit tests focusing on whether loss terms appear when they should. For instance, ensure that when observation_decoder is not provided, the reconstruction loss is not in the output losses dict, etc. Also test that controller.act() works for both types.

Extend to Full Feature Parity: Flesh out the remaining needed functionality for each component type. For instance, if we haven’t implemented MCTS fully in step 4, decide if it’s in scope or leave a stub/planner interface that can be filled later. Similarly, ensure the transformer dynamics can ingest sequence data properly – this might involve modifying how we call dynamics_model.step for transformer (maybe it needs the entire history; we could let it internally store history or require a reset() call at episode start to clear it). Also, incorporate any training tricks: e.g., for RSSM dynamics, incorporate KL balancing or free bits (these can be parameters in config passed to dynamics_loss). Essentially, iterate on each axis to ensure our abstractions aren’t missing a critical feature. Write tests for specific things, e.g., if we set a ShortcutForcing flag on a transformer dynamics, does the loss behave as expected (we might need a mock to test multi-step prediction).

Performance Profiling & Optimization: Once it’s functionally working for a couple of algorithms, consider performance. Profile the imagination rollout, especially for transformer dynamics which might be slower if not batched. We may need to implement batch rollouts (the current rollout_imagination loops step by step in Python
GitHub
 – if using a transformer that attends over the whole sequence, it might be better to generate the whole sequence in one pass instead of step loop). We can introduce an optimization: if dynamics has a method rollout_many_steps(initial_state, policy, horizon) it could override the default loop with something faster internally. For example, a transformer could take an initial sequence and auto-regressively generate next tokens in a single forward pass if implemented cleverly. This can be done later, but it’s something to keep in mind (our interface should allow it, maybe by letting DynamicsCore either handle a multi-step rollout or by vectorizing the loop outside).
Additionally, if we find that the abstraction causes a lot of tiny torch ops (overlooking some fusion), we can refactor accordingly. However, given that we can always drop to lower-level if needed (e.g., for MuZero MCTS we might write parts in C++ for speed but that doesn’t affect the interface), we have flexibility.

Documentation and Examples: Write documentation in the repo (e.g., docs/world_model_paradigm.md or update your existing docs/) explaining the new design. Include a “cookbook” of configurations: show how to configure Dreamer, how to configure MuZero, etc., in YAML. Provide guidance on how to add a new component to the registry (for contributors who might add a new type of dynamics or head). Emphasize the mapping from research papers to config fields (this helps users validate that e.g. discrete_latent: true corresponds to using a VQ-VAE like in IRIS, etc.). Good docs will also help to alleviate the learning curve issue mentioned earlier.

Gradual Migration & Backward Compatibility: If you have existing experiments or users of the old WorldModelParadigm, ensure they still work or have a migration path. One strategy is to keep the old implementation accessible (perhaps under a config flag or as a separate class DreamerParadigm that inherits from WorldModelParadigm but overrides compute_loss to the old behavior) until you’ve switched everything to the new system. Then deprecate the old style. This way current results or models aren’t immediately broken. Since you already noted not everything was built yet in the world model module, the impact might be minimal. But it’s good practice to announce changes in the CHANGELOG or docs.

By following this implementation plan, we ensure that we build the system iteratively and verify each piece. The end result will be an extensible WorldModel framework that captures the similarities of current methods while being able to toggle the differences through configuration.

Conclusion

To solve your problem: we will build a highly modular “World Model Paradigm” system that abstracts the common components (encoder, latent state, dynamics model, predictor heads, controller, etc.) and makes each configurable. This design draws on the insights from SOTA world models:

It will allow continuous vs discrete latents, as inspired by Dreamer vs IRIS
arxiv.org
.

It will permit both learning-by-imagination (Dreamer) and planning (TD-MPC, MuZero) by swapping out the Controller and adjusting loss computation
arxiv.org
arxiv.org
.

It will include or exclude reconstruction and reward prediction losses as needed, accommodating value-focused approaches that skip observation prediction
arxiv.org
.

It will support transformer-based dynamics for scalability (e.g. Dreamer4’s approach)
emergentmind.com
, as well as classic RNN dynamics, under a unified API.

Crucially, it will maintain the “plug-and-play” ethos already present in your PPO code, ensuring researchers can experiment with minimal friction.

By separating axes of variation, we capture the “abstracted similarities” you aimed for. Each new world model architecture can be seen as a different point in this configuration space, rather than a whole new codebase. Initially prioritizing flexibility will make it easier to incorporate efficiency improvements later (for example, once the design is in place, optimizing the transformer rollout or parallelizing MCTS can be tackled knowing the interface won’t change).

 

In sum, the plan is to generalize your WorldModelParadigm into a full-fledged framework for world models. We’ve reviewed Codex’s proposed design against actual research and found it well-aligned: it indeed exposes the key axes and suggests sensible abstractions. We supplemented it with up-to-date context (like Dreamer4’s innovations and TD-MPC2’s scale) to ensure the design will accommodate recent advances. By following the implementation steps with careful testing and documentation, you will arrive at a solution that is both comprehensive and robust: one codebase to rule them all (world models, that is)
arxiv.org
emergentmind.com
. This positions your project to quickly adopt new ideas and consistently benchmark them, fulfilling the objective of a universal world-model architecture.

 

Sources:

Hafner et al., “Dreamer 4: Scalable World Model Agent” – Key idea of transformer-based world model with offline training and imagination
emergentmind.com
emergentmind.com
.

Hansen et al., “TD-MPC2: Scalable, Robust World Models for Continuous Control” – Implicit decoder-free model and latent planning at large scale
arxiv.org
arxiv.org
.

Micheli et al., “IRIS: Transformers are Sample-Efficient World Models” – Discrete autoencoder + Transformer dynamics for data-efficient RL
arxiv.org
arxiv.org
.

Chen et al., “TransDreamer: Reinforcement Learning with Transformer World Models” – Replacing RSSM with transformer, improving long-horizon memory in Dreamer
arxiv.org
arxiv.org
.

Schrittwieser et al., “MuZero”, Nature 2020 – Learning a model that predicts reward, policy, value; planning with MCTS instead of reconstruction
arxiv.org
arxiv.org
.

AdarshGoel2001/rl-lab (GitHub repository) – Current implementation of WorldModelParadigm showing hard-coded Dreamer-style imagination and losses
GitHub
GitHub
, and modular design patterns in the PPO paradigm
GitHub
GitHub
