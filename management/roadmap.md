# World Models Roadmap: Jan 8 - May 31, 2026

## Progress Overview

| Phase | Status | Dates | Completion |
|-------|--------|-------|------------|
| 0: Diffusion Foundations | 🔲 Not Started | Jan 8-21 | 0% |
| 1: Action-Conditioned WM | 🔲 Not Started | Jan 22 - Feb 18 | 0% |
| 2: Diffusion Forcing | 🔲 Not Started | Feb 19 - Mar 4 | 0% |
| 3: VLA Finetuning | 🔲 Not Started | Mar 5 - Apr 1 | 0% |
| 4: Unified World Model | 🔲 Not Started | Apr 2-29 | 0% |
| 5: Integration Project | 🔲 Not Started | Apr 30 - May 27 | 0% |
| Buffer | 🔲 Not Started | May 28-31 | 0% |

**Status Key:** 🔲 Not Started | 🟡 In Progress | ✅ Complete | ⏸️ Paused

---

## Phase 0: Diffusion Foundations (Week 1-2)

### Week 1: DDPM From Scratch (Jan 8-14)
- [ ] Read: DDPM paper
- [ ] Read: Score-Based Generative Models  
- [ ] Read: EDM paper
- [ ] Write: Math notes on diffusion
- [ ] Code: Basic DDPM on MNIST
- [ ] Code: Training loop with logging
- [ ] Code: Sampling (full 1000 steps)
- [ ] Code: EDM formulation rewrite
- [ ] Artifact: Working MNIST diffusion model

### Week 2: CIFAR + CFG (Jan 15-21)
- [ ] Read: Classifier-Free Guidance paper
- [ ] Read: World Models (Ha & Schmidhuber)
- [ ] Read: PlaNet paper (skim)
- [ ] Code: Scale UNet to CIFAR-10
- [ ] Code: Class conditioning
- [ ] Code: CFG implementation
- [ ] Code: DDIM fast sampling
- [ ] Artifact: CIFAR diffusion with CFG
- [ ] Write: Phase 0 README

**Phase 0 Deliverables:**
- [ ] MNIST diffusion model
- [ ] CIFAR diffusion with CFG
- [ ] Notes: `notes/diffusion-fundamentals.md`

---

## Phase 1: Action-Conditioned World Model (Week 3-6)

### Week 3: Environment + Data (Jan 22-28)
- [ ] Read: DIAMOND paper (thorough)
- [ ] Read: Dreamer V1
- [ ] Read: Dreamer V2
- [ ] Setup: Atari environment (gymnasium)
- [ ] Code: Data collection script
- [ ] Collect: 100+ episodes of gameplay
- [ ] Code: Dataset class with frame stacking
- [ ] Verify: Visualize data batches
- [ ] Artifact: Clean data pipeline

### Week 4: Architecture (Jan 29 - Feb 4)
- [ ] Read: IRIS paper
- [ ] Read: STORM paper
- [ ] Read: GameNGen (optional)
- [ ] Design: World model architecture diagram
- [ ] Code: Frame encoder
- [ ] Code: Action conditioning
- [ ] Code: UNet with conditioning
- [ ] Code: Training loop (EDM-style)
- [ ] Run: First training (verify loss decreases)

### Week 5: Training + Rollouts (Feb 5-11)
- [ ] Read: TD-MPC paper
- [ ] Read: TD-MPC2 paper
- [ ] Read: MuZero (skim)
- [ ] Train: Full training run
- [ ] Code: Autoregressive rollout function
- [ ] Evaluate: 10, 20, 50 step rollouts
- [ ] Create: Side-by-side comparison videos
- [ ] Measure: MSE, visual quality metrics

### Week 6: Polish (Feb 12-18)
- [ ] Read: Dreamer V3 (full)
- [ ] Read: Delta-IRIS (skim)
- [ ] Read: TransDreamer (skim)
- [ ] Code: Clean up and document
- [ ] Code: Wandb integration with videos
- [ ] Stretch: Simple policy in imagination
- [ ] Write: Phase 1 README
- [ ] Artifact: Complete DIAMOND-lite

**Phase 1 Deliverables:**
- [ ] Trained world model checkpoint
- [ ] Rollout videos
- [ ] Evaluation metrics
- [ ] Blog post draft: "Building DIAMOND from Scratch"

---

## Phase 2: Diffusion Forcing (Week 7-8)

### Week 7: DF Implementation (Feb 19-25)
- [ ] Read: Diffusion Forcing paper (thorough)
- [ ] Read: Flow Matching paper
- [ ] Read: Rectified Flow (skim)
- [ ] Read: Consistency Models (skim)
- [ ] Write: Notes on Diffusion Forcing algorithm
- [ ] Code: Modify training for independent noise levels
- [ ] Code: Causal masking in model
- [ ] Code: DF sampling procedure
- [ ] Run: Initial training with DF

### Week 8: Comparison (Feb 26 - Mar 4)
- [ ] Read: I-JEPA paper
- [ ] Read: V-JEPA paper
- [ ] Read: LeCun position paper
- [ ] Train: Full training with DF
- [ ] Compare: Standard vs DF rollout quality
- [ ] Measure: Error accumulation over horizon
- [ ] Create: Comparison visualizations
- [ ] Write: Phase 2 analysis document

**Phase 2 Deliverables:**
- [ ] DF-trained world model
- [ ] Comparison experiments
- [ ] Analysis: "Diffusion Forcing: What It Buys You"

---

## Phase 3: VLA Finetuning (Week 9-12)

### Week 9: VLA Setup (Mar 5-11)
- [ ] Read: RT-1 paper
- [ ] Read: RT-2 paper
- [ ] Read: OpenVLA paper (thorough)
- [ ] Read: π0 paper
- [ ] Read: SmolVLA blog/paper
- [ ] Setup: Clone OpenVLA/SmolVLA repos
- [ ] Setup: Get inference working
- [ ] Setup: LIBERO or SimplerEnv
- [ ] Test: Run pretrained model
- [ ] Document: Data format understanding

### Week 10: Data + Finetuning Setup (Mar 12-18)
- [ ] Read: Octo paper
- [ ] Read: Open X-Embodiment
- [ ] Read: DROID paper
- [ ] Read: FAST tokenizer
- [ ] Choose: Task not in training distribution
- [ ] Collect: 50-100 demonstrations
- [ ] Format: Convert to VLA format
- [ ] Setup: LoRA finetuning script
- [ ] Run: First finetuning run

### Week 11: Training + Evaluation (Mar 19-25)
- [ ] Read: Helix paper
- [ ] Read: GR00T N1
- [ ] Read: Gemini Robotics
- [ ] Train: Full LoRA finetuning
- [ ] Evaluate: Success rate on task
- [ ] Compare: Pretrained vs finetuned
- [ ] Record: Rollout videos
- [ ] Analyze: Where does model fail?

### Week 12: Multi-task + Docs (Mar 26 - Apr 1)
- [ ] Read: LIBERO paper
- [ ] Read: V-JEPA 2 (full)
- [ ] Stretch: Multi-task finetuning
- [ ] Ablate: Minimum data needed
- [ ] Ablate: LoRA rank effect
- [ ] Clean: All code
- [ ] Write: Complete documentation
- [ ] Artifact: "Finetuning VLAs: A Practical Guide"

**Phase 3 Deliverables:**
- [ ] Finetuned SmolVLA checkpoint
- [ ] Evaluation results
- [ ] Comparison analysis
- [ ] Blog post / guide

---

## Phase 4: Unified World Model (Week 13-16)

### Week 13: UWM Architecture (Apr 2-8)
- [ ] Read: UWM paper (multiple times)
- [ ] Read: Unified Video Action Model
- [ ] Read: WorldVLA
- [ ] Read: GR-2 paper
- [ ] Write: Architecture design document
- [ ] Code: Video tokenizer
- [ ] Code: Action encoder
- [ ] Code: Transformer backbone
- [ ] Code: Registers
- [ ] Code: Separate heads
- [ ] Code: Training loop with independent timesteps

### Week 14: UWM Training (Apr 9-15)
- [ ] Read: DreamGen paper
- [ ] Read: Video Prediction Policy
- [ ] Read: World Models Survey
- [ ] Train: UWM on LIBERO
- [ ] Monitor: Both modality losses
- [ ] Debug: Ensure both learning
- [ ] Tune: Hyperparameters

### Week 15: Inference Modes (Apr 16-22)
- [ ] Read: Cosmos paper
- [ ] Read: Genie 2 paper
- [ ] Read: Genie 3 blog
- [ ] Code: Policy mode (σ_v=0, denoise actions)
- [ ] Code: Forward dynamics mode (σ_a=0, denoise video)
- [ ] Code: Inverse dynamics mode
- [ ] Evaluate: Each mode quality
- [ ] Compare: To baselines

### Week 16: Experiments + Polish (Apr 23-29)
- [ ] Compare: UWM policy vs SmolVLA
- [ ] Compare: UWM dynamics vs DIAMOND-lite
- [ ] Analyze: Does joint training help?
- [ ] Scale: More data? Larger model?
- [ ] Clean: All code
- [ ] Write: Complete documentation

**Phase 4 Deliverables:**
- [ ] UWM implementation
- [ ] All inference modes working
- [ ] Comparative experiments
- [ ] Full documentation

---

## Phase 5: Integration Project (Week 17-20)

### Week 17: Project Setup (Apr 30 - May 6)
- [ ] Choose: Project option (A/B/C)
- [ ] Define: Specific experiments
- [ ] Define: Success metrics
- [ ] Setup: Evaluation pipeline
- [ ] Setup: Baselines
- [ ] Begin: Core implementation

### Week 18: Core Implementation (May 7-13)
- [ ] Build: Main system functionality
- [ ] Build: Training/evaluation loops
- [ ] Run: First experiments
- [ ] Debug: Iterate

### Week 19: Experiments (May 14-20)
- [ ] Run: All planned experiments
- [ ] Collect: All metrics
- [ ] Create: Visualizations
- [ ] Analyze: What worked? Why?

### Week 20: Documentation (May 21-27)
- [ ] Clean: All code
- [ ] Write: Technical report
- [ ] Write: Blog post
- [ ] Create: Demo video
- [ ] Update: Portfolio

**Phase 5 Deliverables:**
- [ ] Complete system
- [ ] Experimental results
- [ ] Technical write-up
- [ ] Demo video
- [ ] Portfolio-ready presentation

---

## Week 21: Buffer (May 28-31)
- [ ] Finish: Any incomplete items
- [ ] Fix: Bugs found in documentation
- [ ] Write: "What I Learned" reflection
- [ ] Plan: What's next?

---

## Final Artifacts Checklist

- [ ] Phase 0: Diffusion model (MNIST + CIFAR)
- [ ] Phase 1: DIAMOND-lite world model
- [ ] Phase 2: Diffusion Forcing upgrade
- [ ] Phase 3: VLA finetuning pipeline + guide
- [ ] Phase 4: UWM implementation
- [ ] Phase 5: Integration project with results
- [ ] All code documented and clean
- [ ] Blog posts / write-ups
- [ ] Demo videos
- [ ] Portfolio updated
```

---

### 2. GitHub Projects Setup

Create a GitHub Project with these columns:
```
| Backlog | This Week | In Progress | Review | Done |