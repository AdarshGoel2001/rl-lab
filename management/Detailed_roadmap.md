# World Models Implementation Roadmap
## January 8, 2026 → May 31, 2026 (21 weeks)

---

## 📊 Overall Progress

```
Phase 0: Diffusion Foundations    [----------] 0/9 tasks   0%
Phase 1: Action-Conditioned WM    [----------] 0/20 tasks  0%
Phase 2: Diffusion Forcing        [----------] 0/12 tasks  0%
Phase 3: VLA Finetuning           [----------] 0/24 tasks  0%
Phase 4: Unified World Model      [----------] 0/20 tasks  0%
Phase 5: Integration Project      [----------] 0/16 tasks  0%

TOTAL PROGRESS                    [----------] 0/101 tasks 0%
```

**Last Updated:** Not started  
**Current Phase:** Not started  
**Current Week:** Not started  
**Days Until Deadline:** 144 (as of Jan 8)

---

## 📅 Timeline Overview

| Phase | Weeks | Dates | Status | Tasks |
|-------|-------|-------|--------|-------|
| 0: Diffusion Foundations | 1-2 | Jan 8-21 | 🔲 | 0/9 |
| 1: Action-Conditioned WM | 3-6 | Jan 22 - Feb 18 | 🔲 | 0/20 |
| 2: Diffusion Forcing | 7-8 | Feb 19 - Mar 4 | 🔲 | 0/12 |
| 3: VLA Finetuning | 9-12 | Mar 5 - Apr 1 | 🔲 | 0/24 |
| 4: Unified World Model | 13-16 | Apr 2-29 | 🔲 | 0/20 |
| 5: Integration Project | 17-20 | Apr 30 - May 27 | 🔲 | 0/16 |
| Buffer | 21 | May 28-31 | 🔲 | - |

**Status Key:** 🔲 Not Started | 🟡 In Progress | ✅ Complete | ⏸️ Paused | 🔴 Behind

---

## 📚 Reading Schedule by Week

| Week | Papers | Status |
|------|--------|--------|
| 1 | DDPM, Score-Based Models, EDM | 🔲 |
| 2 | CFG, World Models (2018), PlaNet | 🔲 |
| 3 | DIAMOND, Dreamer V1, Dreamer V2 | 🔲 |
| 4 | IRIS, STORM, GameNGen | 🔲 |
| 5 | TD-MPC, TD-MPC2, MuZero | 🔲 |
| 6 | Dreamer V3, Delta-IRIS, TransDreamer | 🔲 |
| 7 | Diffusion Forcing, Flow Matching, Rectified Flow | 🔲 |
| 8 | I-JEPA, V-JEPA, LeCun Position Paper | 🔲 |
| 9 | RT-1, RT-2, OpenVLA, π0, SmolVLA | 🔲 |
| 10 | Octo, Open X-Embodiment, DROID, FAST | 🔲 |
| 11 | Helix, GR00T N1, Gemini Robotics | 🔲 |
| 12 | LIBERO, V-JEPA 2 | 🔲 |
| 13 | UWM, UVAM, WorldVLA, GR-2 | 🔲 |
| 14 | DreamGen, Video Prediction Policy, Survey | 🔲 |
| 15 | Cosmos, Genie 2, Genie 3 | 🔲 |
| 16 | Catch-up, surveys | 🔲 |

---

# PHASE 0: DIFFUSION FOUNDATIONS
**Weeks 1-2 | Jan 8-21 | Goal: Understand diffusion deeply through implementation**

## Week 1: DDPM From Scratch (Jan 8-14)

### Reading
- [ ] DDPM (Ho et al., 2020) - Full read, understand all equations
- [ ] Score-Based Generative Models (Song & Ermon) - Alternative perspective
- [ ] EDM (Karras et al., 2022) - Focus on Section 2, clean formulation

### Implementation
- [ ] Setup repo structure and environment
- [ ] Write math notes explaining forward/reverse process
- [ ] Implement basic UNet architecture
- [ ] Implement DDPM training loop on MNIST
- [ ] Implement full 1000-step sampling
- [ ] Rewrite using EDM formulation
- [ ] Log training to wandb/tensorboard

### Deliverable
- [ ] Working MNIST diffusion model generating recognizable digits

**Week 1 Progress:** 6/12 tasks

---

## Week 2: CIFAR + Classifier-Free Guidance (Jan 15-21)

### Reading
- [ ] Classifier-Free Guidance (Ho & Salimans, 2022)
- [ ] World Models (Ha & Schmidhuber, 2018) - Start understanding "why"
- [ ] PlaNet (Hafner et al., 2019) - Skim

### Implementation
- [ ] Scale UNet to CIFAR-10 (32×32×3)
- [ ] Add class conditioning via embedding
- [ ] Implement CFG (dropout + guidance scale)
- [ ] Implement DDIM for fast sampling
- [ ] Compare: DDPM 1000 steps vs DDIM 50 steps
- [ ] Create comparison visualizations

### Deliverable
- [ ] CIFAR diffusion with CFG, samples at different guidance scales
- [ ] Phase 0 README and notes

**Week 2 Progress:** 0/9 tasks

---

## Phase 0 Summary

**Total Tasks:** 0/18 complete

**Deliverables Checklist:**
- [ ] `phase0-diffusion/` - Working code
- [ ] MNIST model checkpoint
- [ ] CIFAR model checkpoint  
- [ ] `notes/diffusion-fundamentals.md`
- [ ] Phase 0 README

---

# PHASE 1: ACTION-CONDITIONED WORLD MODEL
**Weeks 3-6 | Jan 22 - Feb 18 | Goal: Build DIAMOND-lite**

## Week 3: Environment + Data Pipeline (Jan 22-28)

### Reading
- [ ] DIAMOND (Alonso et al., NeurIPS 2024) - Thorough read
- [ ] Dreamer V1 - Understand "dream" concept
- [ ] Dreamer V2 - Understand discrete representations

### Implementation
- [ ] Setup Atari environment (gymnasium + ale-py)
- [ ] Write data collection script (random policy)
- [ ] Collect 100+ episodes of Breakout/Pong
- [ ] Implement AtariWorldModelDataset with frame stacking
- [ ] Verify: visualize batches, check action alignment

### Deliverable
- [ ] Clean data pipeline with visualization

**Week 3 Progress:** 0/8 tasks

---

## Week 4: World Model Architecture (Jan 29 - Feb 4)

### Reading
- [ ] IRIS - Understand discrete token approach
- [ ] STORM - Transformer dynamics
- [ ] GameNGen - Optional, diffusion for DOOM

### Implementation
- [ ] Design architecture (draw diagram first)
- [ ] Implement frame encoder for context
- [ ] Implement action conditioning mechanism
- [ ] Build UNet with EDM + conditioning
- [ ] Implement training loop
- [ ] First training run - verify loss decreases

### Deliverable
- [ ] Training world model architecture

**Week 4 Progress:** 0/9 tasks

---

## Week 5: Training + Autoregressive Rollout (Feb 5-11)

### Reading
- [ ] TD-MPC - Understand MPC conceptually
- [ ] TD-MPC2 - Scaling insights
- [ ] MuZero - Background on learned dynamics

### Implementation
- [ ] Full training run (50-100 epochs)
- [ ] Implement autoregressive rollout function
- [ ] Generate 10, 20, 50 step rollouts
- [ ] Create side-by-side videos (predicted vs actual)
- [ ] Measure: MSE, perceptual quality

### Deliverable
- [ ] Trained model with evaluation metrics

**Week 5 Progress:** 0/8 tasks

---

## Week 6: Polish + Stretch Goals (Feb 12-18)

### Reading
- [ ] Dreamer V3 (full read) - Robustness techniques
- [ ] Delta-IRIS - Skim
- [ ] TransDreamer - Skim

### Implementation
- [ ] Code cleanup and documentation
- [ ] Wandb integration with video logging
- [ ] (Stretch) Train simple policy in imagination
- [ ] Write Phase 1 README
- [ ] Create blog post draft

### Deliverable
- [ ] Complete DIAMOND-lite implementation
- [ ] Blog: "Building DIAMOND from Scratch"

**Week 6 Progress:** 0/8 tasks

---

## Phase 1 Summary

**Total Tasks:** 0/33 complete

**Deliverables Checklist:**
- [ ] `phase1-world-model/` - Working code
- [ ] Trained model checkpoint
- [ ] Rollout comparison videos
- [ ] Evaluation metrics document
- [ ] Phase 1 README
- [ ] Blog post draft

---

# PHASE 2: DIFFUSION FORCING
**Weeks 7-8 | Feb 19 - Mar 4 | Goal: Modern training paradigm**

## Week 7: Diffusion Forcing Implementation (Feb 19-25)

### Reading
- [ ] Diffusion Forcing (Chen et al., NeurIPS 2024) - Thorough
- [ ] Flow Matching (Lipman et al., 2022)
- [ ] Rectified Flow - Skim
- [ ] Consistency Models - Skim

### Implementation
- [ ] Write detailed notes on DF algorithm
- [ ] Modify training for independent noise levels per frame
- [ ] Implement causal masking
- [ ] Implement DF sampling procedure
- [ ] Initial training run with DF

### Deliverable
- [ ] DF training working

**Week 7 Progress:** 0/9 tasks

---

## Week 8: Comparison + Analysis (Feb 26 - Mar 4)

### Reading
- [ ] I-JEPA - Embedding-space prediction
- [ ] V-JEPA - Video extension
- [ ] LeCun Position Paper - "A Path Towards AMI"

### Implementation
- [ ] Full training with Diffusion Forcing
- [ ] Compare: Standard autoregressive vs DF
- [ ] Measure: Error accumulation over horizon
- [ ] Create comparison visualizations
- [ ] Write analysis document

### Deliverable
- [ ] "Diffusion Forcing: What It Buys You" analysis

**Week 8 Progress:** 0/8 tasks

---

## Phase 2 Summary

**Total Tasks:** 0/17 complete

**Deliverables Checklist:**
- [ ] `phase2-diffusion-forcing/` - Working code
- [ ] DF-trained model checkpoint
- [ ] Comparison experiments results
- [ ] Analysis document
- [ ] Phase 2 README

---

# PHASE 3: VLA FINETUNING
**Weeks 9-12 | Mar 5 - Apr 1 | Goal: Practical VLA skills**

## Week 9: VLA Foundations + Setup (Mar 5-11)

### Reading
- [ ] RT-1 - Original robotics transformer
- [ ] RT-2 - VLM → VLA transition
- [ ] OpenVLA - Architecture deep dive
- [ ] π0 - Flow matching for actions
- [ ] SmolVLA - Efficient VLA

### Implementation
- [ ] Clone and setup OpenVLA/SmolVLA repos
- [ ] Get pretrained inference working
- [ ] Setup LIBERO or SimplerEnv
- [ ] Test pretrained model in simulation
- [ ] Document data format understanding

### Deliverable
- [ ] Working VLA inference setup

**Week 9 Progress:** 0/10 tasks

---

## Week 10: Data Collection + Finetuning Setup (Mar 12-18)

### Reading
- [ ] Octo - Cross-embodiment
- [ ] Open X-Embodiment - Datasets
- [ ] DROID - Real-world data
- [ ] FAST tokenizer

### Implementation
- [ ] Choose task NOT in training distribution
- [ ] Collect 50-100 demonstrations
- [ ] Format data into VLA structure
- [ ] Setup LoRA finetuning script
- [ ] First finetuning run (verify it works)

### Deliverable
- [ ] Finetuning pipeline ready

**Week 10 Progress:** 0/9 tasks

---

## Week 11: Training + Evaluation (Mar 19-25)

### Reading
- [ ] Helix (Figure AI) - Dual-system VLA
- [ ] GR00T N1 (NVIDIA) - Humanoid VLA
- [ ] Gemini Robotics - If available

### Implementation
- [ ] Full LoRA finetuning run
- [ ] Evaluate success rate
- [ ] Compare pretrained vs finetuned
- [ ] Record rollout videos
- [ ] Analyze failure cases

### Deliverable
- [ ] Finetuned model with evaluation

**Week 11 Progress:** 0/8 tasks

---

## Week 12: Multi-task + Documentation (Mar 26 - Apr 1)

### Reading
- [ ] LIBERO benchmark paper
- [ ] V-JEPA 2 (full read)

### Implementation
- [ ] (Stretch) Multi-task finetuning on 3-5 tasks
- [ ] Ablation: Minimum data needed
- [ ] Ablation: LoRA rank effect
- [ ] Complete code cleanup
- [ ] Write comprehensive documentation
- [ ] Create "Finetuning VLAs" guide

### Deliverable
- [ ] Complete VLA finetuning pipeline + guide

**Week 12 Progress:** 0/8 tasks

---

## Phase 3 Summary

**Total Tasks:** 0/35 complete

**Deliverables Checklist:**
- [ ] `phase3-vla-finetuning/` - Working code
- [ ] Finetuned SmolVLA checkpoint
- [ ] Evaluation results
- [ ] Rollout videos
- [ ] "Finetuning VLAs: A Practical Guide"
- [ ] Phase 3 README

---

# PHASE 4: UNIFIED WORLD MODEL
**Weeks 13-16 | Apr 2-29 | Goal: Implement frontier architecture**

## Week 13: UWM Architecture Study (Apr 2-8)

### Reading
- [ ] UWM (Zhu et al., RSS 2025) - Multiple reads
- [ ] Unified Video Action Model
- [ ] WorldVLA
- [ ] GR-2

### Implementation
- [ ] Write architecture design document
- [ ] Implement video tokenizer
- [ ] Implement action encoder
- [ ] Implement transformer backbone
- [ ] Implement registers
- [ ] Implement separate heads
- [ ] Implement training loop with independent timesteps

### Deliverable
- [ ] UWM architecture ready for training

**Week 13 Progress:** 0/11 tasks

---

## Week 14: UWM Training (Apr 9-15)

### Reading
- [ ] DreamGen
- [ ] Video Prediction Policy
- [ ] World Models Survey (Ding et al.)

### Implementation
- [ ] Train UWM on LIBERO
- [ ] Monitor both modality losses
- [ ] Debug and ensure both learning
- [ ] Hyperparameter tuning

### Deliverable
- [ ] Training UWM model

**Week 14 Progress:** 0/7 tasks

---

## Week 15: Flexible Inference Modes (Apr 16-22)

### Reading
- [ ] NVIDIA Cosmos
- [ ] Genie 2
- [ ] Genie 3

### Implementation
- [ ] Implement policy mode (σ_v=0, denoise actions)
- [ ] Implement forward dynamics mode (σ_a=0, denoise video)
- [ ] Implement inverse dynamics mode
- [ ] Evaluate each mode
- [ ] Compare to baselines

### Deliverable
- [ ] All inference modes working

**Week 15 Progress:** 0/8 tasks

---

## Week 16: Experiments + Polish (Apr 23-29)

### Implementation
- [ ] Compare UWM policy vs SmolVLA
- [ ] Compare UWM dynamics vs DIAMOND-lite
- [ ] Analyze: Does joint training help?
- [ ] Scale experiments (more data, larger model)
- [ ] Code cleanup
- [ ] Complete documentation

### Deliverable
- [ ] Complete UWM with experiments

**Week 16 Progress:** 0/6 tasks

---

## Phase 4 Summary

**Total Tasks:** 0/32 complete

**Deliverables Checklist:**
- [ ] `phase4-uwm/` - Working code
- [ ] UWM model checkpoint
- [ ] All inference modes working
- [ ] Comparative experiments
- [ ] Phase 4 README

---

# PHASE 5: INTEGRATION PROJECT
**Weeks 17-20 | Apr 30 - May 27 | Goal: Complete system**

## Week 17: Project Definition (Apr 30 - May 6)

### Project Options (Choose ONE):
- [ ] **Option A:** World Model Augmented VLA
- [ ] **Option B:** Planning with World Models  
- [ ] **Option C:** Cross-Task Generalization Study

### Implementation
- [ ] Choose project option
- [ ] Define specific experiments
- [ ] Define success metrics
- [ ] Setup evaluation pipeline
- [ ] Setup baselines
- [ ] Begin core implementation

**Week 17 Progress:** 0/6 tasks

---

## Week 18: Core Implementation (May 7-13)

### Implementation
- [ ] Core system functionality
- [ ] Training/evaluation loops
- [ ] First experiments
- [ ] Debug and iterate

**Week 18 Progress:** 0/4 tasks

---

## Week 19: Experiments + Results (May 14-20)

### Implementation
- [ ] Run all planned experiments
- [ ] Collect all metrics
- [ ] Create visualizations
- [ ] Analyze results
- [ ] Document findings

**Week 19 Progress:** 0/5 tasks

---

## Week 20: Documentation + Presentation (May 21-27)

### Implementation
- [ ] Code cleanup and docstrings
- [ ] Write technical report
- [ ] Write blog post
- [ ] Create demo video
- [ ] Update portfolio

**Week 20 Progress:** 0/5 tasks

---

## Week 21: Buffer (May 28-31)

- [ ] Finish any incomplete items
- [ ] Fix documentation bugs
- [ ] Write reflection: "What I Learned"
- [ ] Plan: What's next?

**Week 21 Progress:** 0/4 tasks

---

## Phase 5 Summary

**Total Tasks:** 0/24 complete

**Deliverables Checklist:**
- [ ] `phase5-integration/` - Working code
- [ ] Complete system
- [ ] Experimental results
- [ ] Technical write-up
- [ ] Blog post
- [ ] Demo video
- [ ] Portfolio updated

---

# FINAL DELIVERABLES

## Code Artifacts
- [ ] Phase 0: Diffusion models (MNIST + CIFAR)
- [ ] Phase 1: DIAMOND-lite world model
- [ ] Phase 2: Diffusion Forcing upgrade
- [ ] Phase 3: VLA finetuning pipeline
- [ ] Phase 4: UWM implementation
- [ ] Phase 5: Integration project

## Written Artifacts
- [ ] Blog: "Building DIAMOND from Scratch"
- [ ] Analysis: "Diffusion Forcing: What It Buys You"
- [ ] Guide: "Finetuning VLAs Practically"
- [ ] Technical Report: Integration Project
- [ ] Reflection: "What I Learned Building World Models"

## Media Artifacts
- [ ] Demo videos for each phase
- [ ] Comparison visualizations
- [ ] Portfolio presentation

---

# COMPLETION LOG

Record completions here for satisfaction:

| Date | What I Completed | Phase |
|------|------------------|-------|
| | | |
| | | |
| | | |

---

*This document is the single source of truth. Update it weekly.*