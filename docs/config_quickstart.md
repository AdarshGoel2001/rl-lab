# Config Fix - Quick Start Guide

## 🚀 TL;DR - What You Need to Do NOW

### Step 1: Clean Your Config (30 seconds)

Edit your config file:
```bash
nano configs/experiments/world_model_cartpole_mvp_stable.yaml
```

**Delete or comment out** the entire `paradigm_config` section (lines 86-102):
```yaml
# DELETE THIS ENTIRE SECTION:
# paradigm_config:
#   implementation: world_model_mvp
#   world_model_lr: 2e-4
#   ...
```

Your `algorithm` section already has the correct values - just remove `paradigm_config`!

### Step 2: Validate (10 seconds)

```bash
python scripts/validate_config.py \
    --config configs/experiments/world_model_cartpole_mvp_stable.yaml \
    --expected imagination_horizon=10 entropy_coef=0.001
```

**Look for:** All green checkmarks ✓

### Step 3: Run Training

```bash
python scripts/train.py --config configs/experiments/world_model_cartpole_mvp_stable.yaml
```

**Look for in logs:**
```
Loading paradigm configuration from 'algorithm' section
imagination_horizon            = 10
entropy_coef                   = 1.00e-03
```

**If you see `imagination_horizon = 1`, STOP and re-check Step 1!**

## 📊 What to Expect

### Performance Comparison

| Metric | Old (horizon=1) | New (horizon=10) |
|--------|----------------|------------------|
| Final Return | 34-69 | **200-400** ⬆️ |
| Policy Entropy | 0.693 (stuck) | **0.3-0.5** ⬇️ |
| Breakthrough | ~60k steps | **<30k steps** ⬆️ |

### Training Timeline (200k steps, ~5 minutes)

**With horizon=10:**
- Steps 0-25k: Warmup, random exploration (~20 return)
- Steps 25k-50k: **Breakthrough** to 100+ return
- Steps 50k-100k: Stabilization to 200-300 return
- Steps 100k-200k: Refinement to 300-400 return

**You'll know it's working when:**
- ✅ Deterministic return > 100 by step 50k
- ✅ Stochastic return improving (not stuck at 19)
- ✅ Entropy decreasing from 0.693 toward 0.3-0.5

## 🔍 Troubleshooting

### Problem: Still seeing `imagination_horizon = 1`

**Solution:**
```bash
# Check if paradigm_config exists
grep -A 20 "paradigm_config:" configs/experiments/world_model_cartpole_mvp_stable.yaml

# If you see output, delete those lines
# Then re-validate
```

### Problem: Validation script shows mismatches

**Example:**
```
✗ imagination_horizon = 1  (expected: 10, MISMATCH!)
```

**Solution:** The `paradigm_config` section still exists. Delete it completely.

### Problem: Training shows warnings

**Example:**
```
⚠️  imagination_horizon=1 detected! This severely limits world model planning.
```

**Solution:** Config wasn't properly updated. Go back to Step 1.

## 📝 One-Line Fix

If you just want to test immediately:

```bash
# Backup your config
cp configs/experiments/world_model_cartpole_mvp_stable.yaml configs/experiments/world_model_cartpole_mvp_stable.yaml.bak

# Remove paradigm_config section (lines 86-102 based on current file)
sed -i '86,102d' configs/experiments/world_model_cartpole_mvp_stable.yaml

# Validate
python scripts/validate_config.py --config configs/experiments/world_model_cartpole_mvp_stable.yaml

# Train
python scripts/train.py --config configs/experiments/world_model_cartpole_mvp_stable.yaml
```

## 🎯 Success Checklist

Before running training, verify:

- [ ] `paradigm_config` section removed from YAML
- [ ] Validation script shows ✓ for all expected values
- [ ] Training logs show "Loading paradigm configuration from 'algorithm' section"
- [ ] Training logs show `imagination_horizon = 10`
- [ ] Training logs show `entropy_coef = 1.00e-03`

Within first 50k steps, verify:

- [ ] Deterministic return > 100
- [ ] No warnings about `imagination_horizon=1`
- [ ] Entropy decreasing (check training logs or tensorboard)

---

**Still stuck?** Run the validation script and check `docs/config_fix_summary.md` for detailed explanation.
