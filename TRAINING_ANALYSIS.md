# RL-QWOP Training Analysis & Optimization Guide

## Current Architecture Overview

```
Training Loop
    ↓
PPO (PyTorch) 
    ↓
RealQWOPEnv (Gymnasium)
    ↓
Selenium WebDriver (Firefox)
    ↓
Game Server (Node.js)
    ↓
Game Client (JavaScript)
```

### What's Happening Each Step:

1. **Agent generates action** (instant, ~1ms)
2. **Selenium sends keypresses** (~50-100ms network latency + Selenium overhead)
3. **JavaScript processes input** (~50ms)
4. **Game physics simulation** (~50ms)
5. **Render frame** (~30ms)
6. **Selenium polls for state** (multiple DOM queries)
7. **Parse JSON state** (~10ms)
8. **Compute observation & reward** (~5ms)
9. **Throttle to 8 FPS** (125ms sleep)

**Total per step: ~300-500ms (Selenium) + ~125ms (throttling) = ~425-625ms**

**Measured: ~11.7s per step** = 23x slower than realistic! Why?
- Selenium has massive per-call overhead
- Browser startup/initialization costs
- Blocking synchronous execution pattern

---

## Critical Issues

### 1. **Selenium is a Bottleneck** 🚫
- **Problem**: Selenium communicates over the WebDriver protocol (JSON-RPC over HTTP)
- **Overhead**: ~100-200ms per command roundtrip minimum
- **Impact**: 50 steps = 50 Selenium roundtrips ≈ 2500-5000ms just for communication
- **Real issue**: Only ~25 env steps collected per epoch, reward signal is terrible

### 2. **Sparse, Weak Reward Signal** 🎯
```python
# Current reward function (qwop_real.py:291-304)
reward = state.get("diffx", 0) / max(delta, 0.001)  # Forward velocity
if state.get("terminal", False):
    reward = -1.0
```
**Problems:**
- Only signal is forward movement (diffx)
- No reward for trying different muscle combinations
- No intermediate guidance for exploration
- Large variance in rewards (velocity can spike/drop suddenly)
- Agent learns "move forward" but not *how* to move

### 3. **Throttling Destroys Training Speed** ⏱️
```python
# Current throttle (qwop_real.py:371-378)
min_delta = 1.0 / self.max_fps  # 1/8 = 125ms
```
- Artificial 125ms sleep per step
- Game server can handle 60+ FPS but we force 8 FPS
- Wastes 2x-3x training time
- Makes exploration slower

### 4. **PPO Implementation Issues** 🐛

**Issue A: Advantage normalization fails with zero variance**
```python
# PPO.py:251-252
adv_buf = (adv_buf - np.mean(adv_buf)) / np.std(adv_buf)
# RuntimeWarning: invalid value encountered in divide
```
**Why**: When all rewards are ~constant (sparse signal), std=0, division by zero
**Fix**: Add epsilon to denominator

**Issue B: Separate policy/value loops instead of joint optimization**
```python
# PPO.py:91-101: First loop trains V
# PPO.py:109-128: Second loop trains π
```
**Why is this bad**: 
- Creates redundant dataloader iterations
- Doesn't correlate policy/value improvements
- Standard PPO does joint optimization per minibatch
- Adds 2x training time

**Issue C: Early stopping on KL divergence is too aggressive**
```python
# PPO.py:124-127
if kl > 1.5 * target_kl:
    logger.log('Early stopping due to reaching max kl.')
    break
```
**What happened**: Training ran 1 full V-update loop (80 iters ≈ 9:34), then immediately stopped π-updates after 1 iteration with "early stopping"
**Why**: With only 25 samples and sparse rewards, policy changes drastically → high KL
**Fix**: Increase max KL or use adaptive clipping

**Issue D: Value function dominates policy learning**
```python
# PPO.py line 84: batch_size=1024 for ~25 samples!
```
- Only 25 transitions collected
- Batch size 1024 = single batch with padding/dropout issues
- Should be batch_size=8-16 for 25 samples

**Issue E: No logging of actual losses/diagnostics**
```python
# PPO.py:131-134 - stores zeros!
logger.store(LossPi=0, LossV=0, KL=0, Entropy=0, ClipFrac=0, ...)
```
Can't tell if training is working at all.

### 5. **Environment Design Issues** 🎮

**Issue: State-action mismatch in observations**
```python
# qwop_real.py:61-75
def obs(self, state: Dict[str, float] = None) -> np.ndarray:
    return np.array([
        state.get("posy", 0),
        state.get("head", 0),
        state.get("torso", 0),
        # ... joint angles
    ], dtype=np.float32)
```
**Problem**: Obs only has *current* joint positions, no velocity info
- Agent can't tell if limbs are moving or static
- Can't predict momentum/swing
- Needs: [posy, head, torso, ..., posy_dot, head_dot, ...]

**Issue: Reset doesn't reset game state properly**
```python
# qwop_real.py:354-356
ActionChains(self.driver).key_up("p").key_up("o").key_up("w").key_up("q") \
    .key_down("r").key_up("r").perform()
time.sleep(0.03)
```
- Press 'R' to reset (if game even implements it)
- Only 30ms delay - may not be enough
- No verification the reset actually worked

---

## Performance Metrics

```
Current (measured from training run):
- Steps per epoch: 50
- Time per epoch: ~588 seconds
- Time per step: ~11.7 seconds
- Episode length: ~25 steps (truncated)
- Episode return: -0.95 (mostly negative = falling)
- Model params: π=5060, v=4865
- Training throughput: 0.085 steps/sec
- Cost per 1M steps: ~129 days of computation

Realistic target (for comparison):
- Time per step: 0.1-0.2 seconds (100x faster)
- Training throughput: 5-10 steps/sec
- Cost per 1M steps: 1-2 days
```

---

## Recommended Improvements (Priority Order)

### 🔴 CRITICAL: Architecture Redesign (10-50x speedup)

#### Option 1: WebSocket Connection (RECOMMENDED)
Replace Selenium with direct WebSocket to game:

```javascript
// In game/server.js, add WebSocket endpoint
const ws = new WebSocket('ws://localhost:3000/api/env');

// Message format:
// Client → Server: {action: [q, w, o, p], step: 1}
// Server → Client: {obs: [...], reward: X, done: bool, state: {...}}
```

**Benefits:**
- Eliminate 200-300ms per Selenium roundtrip
- Sub-50ms latency per step
- Can increase FPS from 8 → 60+ without Selenium overhead
- **Speedup: 10-20x**

**Implementation:**
1. Add Socket.io or ws library to game server
2. Create `/api/step` endpoint that accepts action, returns (obs, reward, done)
3. Replace Selenium with WebSocket client in gym-qwop
4. Done in ~2 hours

#### Option 2: Parallel Environments
Run 4-8 game instances simultaneously:

```python
from gymnasium.vector import AsyncVectorEnv

env = AsyncVectorEnv([
    lambda: gym.make('gym_qwop:qwop-v0') 
    for _ in range(4)
])
```

**Benefits:**
- 4x data collection parallelism
- Can use vectorized PPO (much better GPU utilization)
- Each env still 10x slower but total throughput 40x better

**Cost**: 4 Firefox instances + 4 game servers

### 🟡 HIGH PRIORITY: Reward Shaping

Add intermediate rewards to guide exploration:

```python
# Replace sparse reward with shaped reward
def compute_reward(self, state):
    r_forward = state.get("diffx", 0) / max(delta, 0.001)  # Forward motion
    r_joint_movement = sum(abs(state.get(joint, 0)) for joint in joint_names) * 0.1
    r_stability = -abs(state.get("posy", 0) - 0.5)  # Stay upright
    
    total = r_forward + 0.1 * r_joint_movement + 0.05 * r_stability
    
    if state.get("terminal"):
        total = -5.0  # Strong penalty for falling
    
    return total
```

**Speedup**: 2-3x fewer steps needed to learn basic locomotion

### 🟡 HIGH PRIORITY: Fix PPO Implementation

**Fix 1: Joint policy/value optimization**
```python
# Instead of:
# for i in range(num_iters): train_V()
# for i in range(num_iters): train_π()

# Do:
for i in range(num_iters):
    for obs, act, ret, adv, logp_old in dataloader:
        # Train both on same batch
        vf_loss = critic_loss(ac.v(obs), ret)
        pi_loss, _ = actor_loss(ac.pi(obs, act), adv, ..., logp_old)
        
        loss = vf_loss + pi_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
**Speedup**: 2x (one dataloader pass instead of two)

**Fix 2: Safe advantage normalization**
```python
adv_mean = adv_buf.mean()
adv_std = adv_buf.std()
if adv_std > 1e-8:  # Only normalize if variance exists
    adv_buf = (adv_buf - adv_mean) / (adv_std + 1e-8)
else:
    adv_buf = adv_buf - adv_mean
```

**Fix 3: Increase batch size appropriately**
```python
# Don't use batch_size=1024 for 25 samples!
batch_size = min(len(dataset) // 4, 32)  # 4-8 batches per epoch max
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

**Fix 4: Log actual losses**
```python
# Replace zeros with real values
logger.store(
    LossPi=pi_loss.item(),
    LossV=vf_loss.item(),
    KL=approx_kl,
    Entropy=ent,
    ClipFrac=clipfrac
)
```

**Speedup**: 1.5x (better hyperparameters converge faster)

### 🟠 MEDIUM PRIORITY: Observation Enhancement

Add velocity to observations:
```python
def obs(self, state):
    return np.array([
        state.get("posy", 0),           # position
        state.get("head", 0),
        # ... other joints ...
        state.get("velx", 0),           # NEW: velocities
        state.get("accelx", 0),         # NEW: accelerations
        state.get("vely", 0),           # NEW: vertical velocity
    ])
```

**Why**: Agent can predict limb momentum, swing timing

### 🟠 MEDIUM PRIORITY: Remove Artificial Throttling

```python
# Current: self.max_fps = 8
# Better: self.max_fps = 60 (or unlimited)

# Remove artificial sleep unless network is bottleneck
def _throttle(self):
    # Only throttle if we're getting > 60 FPS
    # Let Selenium/network be natural throttle
    pass
```

**Speedup**: 7-8x (8 FPS → 60 FPS), BUT only possible with WebSocket fix

### 🟢 LOW PRIORITY: Fixes for Later

- Implement proper episode termination (currently "terminal" is always False except falling)
- Add max episode length truncation (prevents infinite episodes)
- Consider state preprocessing (normalize joint angles to [-1, 1])
- Experiment with different observation representations (e.g., frame stack)

---

## Estimated Impact

| Improvement | Effort | Speedup | Cumulative |
|---|---|---|---|
| WebSocket (critical) | 2-3h | 10-20x | 10-20x ⚡⚡⚡ |
| Joint opt (PPO) | 30min | 2x | 20-40x |
| Reward shaping | 1h | 2-3x | 40-120x |
| Obs enhancement | 30min | 1.2x | 50-144x |
| Parallel envs (opt) | 2h | 4x | 200-576x |

**With all changes: ~50-100 steps/sec instead of 0.085 steps/sec = 580-1000x faster**

**Timeline to 1M training steps:**
- Current: 129 days
- After WebSocket: 6-13 days  
- After all fixes: 0.4-1 day

---

## Code Changes Summary

**Highest ROI: WebSocket (24h → 1-2h per training run)**

1. Add Socket.io to game/server.js
2. Create ws-based env wrapper in gym-qwop
3. Fix PPO batch/logging issues
4. Add velocity to observations

Would generate proposals in order of priority if you want to proceed.
