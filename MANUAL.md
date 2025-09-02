# Flappy Bird RL — Step‑by‑Step Manual (From Q‑Learning to DQN)

A practical, from‑scratch guide to wire your own Flappy Bird into an RL training loop, first with **Q‑learning** (tabular, via discretization), then with a **DQN** (neural network approximator). Minimal assumptions, camelCase naming.

---

## Table of Contents

* [Prerequisites](#prerequisites)
* [Step 0: Define a Minimal Env API](#step-0-define-a-minimal-env-api)
* [Step 1: Design the State](#step-1-design-the-state)
* [Step 2: Design the Action Space](#step-2-design-the-action-space)
* [Step 3: Design the Reward](#step-3-design-the-reward)
* [Phase A — Q‑Learning (Tabular)](#phase-a--q-learning-tabular)

  * [A1: Discretize the State](#a1-discretize-the-state)
  * [A2: Initialize the Q‑Table](#a2-initialize-the-q-table)
  * [A3: Epsilon‑Greedy Policy](#a3-epsilon-greedy-policy)
  * [A4: Q‑Learning Update](#a4-q-learning-update)
  * [A5: Training Loop](#a5-training-loop)
  * [A6: Evaluate & Debug](#a6-evaluate--debug)
* [Phase B — DQN (Deep Q‑Network)](#phase-b--dqn-deep-q-network)

  * [B1: Network Architecture](#b1-network-architecture)
  * [B2: Replay Buffer](#b2-replay-buffer)
  * [B3: Target Network](#b3-target-network)
  * [B4: Bellman Targets & Loss](#b4-bellman-targets--loss)
  * [B5: DQN Training Loop](#b5-dqn-training-loop)
  * [B6: Hyperparameters](#b6-hyperparameters)
  * [B7: Stability Tricks](#b7-stability-tricks)
  * [B8: Evaluation Protocol](#b8-evaluation-protocol)
* [Common Pitfalls & Fixes](#common-pitfalls--fixes)
* [Suggested Experiments](#suggested-experiments)
* [Appendix: Math & Pseudocode](#appendix-math--pseudocode)

---

## Prerequisites

* You already have a working Flappy Bird game loop (rendering optional during training).
* Comfortable with Python and basic linear algebra.
* Libraries (for DQN phase): `numpy`, `torch` (or `tensorflow`), `collections`.

> Tip: Keep the game logic untouched; add a thin RL adapter around it.

---

## Step 0: Define a Minimal Env API

Create a Gym‑like wrapper around your game. Keep it tiny and deterministic.

**Interface**

* `reset() -> state`
* `step(action) -> (nextState, reward, done, info)`
* `seed(optional)` for reproducibility.

**Skeleton**

```python
class FlappyEnv:
    def __init__(self, render=False):
        self.render = render
        self.rng = np.random.default_rng(123)

    def reset(self):
        self._resetGameObjects()
        state = self._getState()
        return state

    def step(self, action):  # action: 0=doNothing, 1=flap
        self._applyAction(action)
        self._updatePhysics()
        reward = self._computeReward()
        done = self._isTerminal()
        nextState = self._getState()
        info = {}
        if self.render:
            self._renderOnce()
        return nextState, reward, done, info
```

**Checklist**

* Deterministic physics given same random seed.
* `reset()` returns the *same* shape/type of state every time.
* `step()` advances exactly one tick/frame.

---

## Step 1: Design the State

Keep it small, informative, and Markovian enough.

**Recommended features**

* `birdY` (vertical position)
* `birdVelocityY`
* `pipeDeltaX` (horizontal distance to next pipe’s leading edge)
* `pipeGapCenterY` (vertical center of the gap in next pipe)

**State vector**

```python
state = np.array([
    birdY,
    birdVelocityY,
    pipeDeltaX,
    pipeGapCenterY
], dtype=np.float32)
```

**Normalization (optional)**

* Divide positions by screenHeight.
* Divide velocities by a typical max speed.

---

## Step 2: Design the Action Space

* Discrete actions:

  * `0` → doNothing
  * `1` → flap

Keep actions minimal; Flappy Bird doesn’t need more.

---

## Step 3: Design the Reward

Start simple; tune later.

**Baseline shaping**

* `+1.0` per time step survived (or `+0.1` if you prefer smaller scales).
* `+10.0` when successfully passing a pipe.
* `-100.0` on crash.

**Notes**

* Large negative on crash accelerates learning.
* Time‑step reward prevents trivial “stall” behaviors.

---

## Phase A — Q‑Learning (Tabular)

Use only if you’re okay discretizing the continuous state. Great for understanding.

### A1: Discretize the State

Define bin sizes to map continuous values → integer bins.

```python
def makeDiscretizer(screenHeight, maxVelocityY, maxPipeDeltaX):
    def discretize(state):
        birdY, birdVelocityY, pipeDeltaX, pipeGapCenterY = state
        return (
            int(birdY // 10),                        # 10 px bins
            int((birdVelocityY + maxVelocityY) // 2),# 2 px/s bins, shifted to be >=0
            int(pipeDeltaX // 10),                   # 10 px bins
            int(pipeGapCenterY // 10)                # 10 px bins
        )
    return discretize
```

**Guidelines**

* Fewer bins → faster learning but coarser policy.
* More bins → larger Q‑table and slower learning.

### A2: Initialize the Q‑Table

Use a dictionary of `stateKey -> qValuesForActions`.

```python
qTable = {}
actions = [0, 1]

def getQ(stateKey):
    if stateKey not in qTable:
        qTable[stateKey] = np.zeros(len(actions), dtype=np.float32)
    return qTable[stateKey]
```

### A3: Epsilon‑Greedy Policy

```python
epsilon = 1.0
epsilonMin = 0.01
epsilonDecay = 0.995

def selectAction(stateKey):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    return int(np.argmax(getQ(stateKey)))
```

### A4: Q‑Learning Update

For transition `(s, a, r, s')`:

* Update rule:
  `q(s,a) ← q(s,a) + alpha * (r + gamma * max_a' q(s',a') − q(s,a))`

```python
alpha = 0.1
gamma = 0.99

def updateQ(stateKey, action, reward, nextStateKey, done):
    qValues = getQ(stateKey)
    tdTarget = reward
    if not done:
        tdTarget += gamma * np.max(getQ(nextStateKey))
    tdError = tdTarget - qValues[action]
    qValues[action] += alpha * tdError
```

### A5: Training Loop

```python
for episode in range(numEpisodes):
    state = env.reset()
    stateKey = discretize(state)
    episodeReturn = 0.0
    done = False

    while not done:
        action = selectAction(stateKey)
        nextState, reward, done, info = env.step(action)
        nextStateKey = discretize(nextState)

        updateQ(stateKey, action, reward, nextStateKey, done)

        stateKey = nextStateKey
        episodeReturn += reward

    epsilon = max(epsilonMin, epsilon * epsilonDecay)
    logEpisode(episode, episodeReturn)
```

**When does it work?**

* With *small* maps and coarse bins.
* Good for learning mechanics, not for best performance.

### A6: Evaluate & Debug

* Freeze exploration: `epsilon = 0.0` → pure greedy.
* Track metrics: moving average of episodeReturn.
* Visualize some episodes with rendering.

---

## Phase B — DQN (Deep Q‑Network)

Move to function approximation for continuous state → robust learning.

### B1: Network Architecture

Keep it small; Flappy Bird is simple.

```python
import torch
import torch.nn as nn

class DqnNet(nn.Module):
    def __init__(self, inputDim=4, outputDim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputDim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, outputDim)
        )
    def forward(self, x):
        return self.model(x)
```

### B2: Replay Buffer

Stores transitions for off‑policy updates.

```python
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', 'state action reward nextState done')

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batchSize):
        return random.sample(self.buffer, batchSize)
    def __len__(self):
        return len(self.buffer)
```

### B3: Target Network

Stabilizes training by decoupling target computation.

```python
targetUpdateInterval = 1000  # steps
policyNet = DqnNet()
targetNet = DqnNet()
targetNet.load_state_dict(policyNet.state_dict())
for p in targetNet.parameters():
    p.requires_grad_(False)
```

### B4: Bellman Targets & Loss

For a batch `(s, a, r, s', d)`:

* Targets: `y = r + gamma * (1 - d) * max_a' Q_target(s', a')`
* Loss: MSE between predicted `Q_policy(s, a)` and `y`.

```python
optimizer = torch.optim.Adam(policyNet.parameters(), lr=1e-3)
criterion = nn.MSELoss()
```

### B5: DQN Training Loop

```python
gamma = 0.99
epsilon = 1.0
epsilonMin = 0.05
epsilonDecay = 0.995
batchSize = 64
warmupSteps = 1000
trainInterval = 4
stepCount = 0

replayBuffer = ReplayBuffer(50000)

state = env.reset()
for episode in range(numEpisodes):
    state = env.reset()
    done = False
    while not done:
        stepCount += 1
        # ε-greedy action
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 2)
        else:
            with torch.no_grad():
                q = policyNet(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(q, dim=1).item())

        nextState, reward, done, info = env.step(action)
        replayBuffer.push(state, action, reward, nextState, float(done))

        state = nextState

        # Learn
        if len(replayBuffer) >= warmupSteps and stepCount % trainInterval == 0:
            batch = replayBuffer.sample(batchSize)
            s, a, r, ns, d = zip(*batch)
            s = torch.tensor(np.array(s), dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
            r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
            ns = torch.tensor(np.array(ns), dtype=torch.float32)
            d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

            qSa = policyNet(s).gather(1, a)
            with torch.no_grad():
                qNextMax = targetNet(ns).max(dim=1, keepdim=True)[0]
                y = r + gamma * (1.0 - d) * qNextMax

            loss = criterion(qSa, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policyNet.parameters(), 10.0)
            optimizer.step()

        # Target network update
        if stepCount % targetUpdateInterval == 0:
            targetNet.load_state_dict(policyNet.state_dict())

    epsilon = max(epsilonMin, epsilon * epsilonDecay)
    logEpisode(episode, loss=float(loss.item()) if 'loss' in locals() else None)
```

### B6: Hyperparameters

* `gamma`: `0.99`
* `learningRate`: `1e-3` (try `5e-4`–`1e-3`)
* `batchSize`: `64` (try `32–128`)
* `replayCapacity`: `50k–200k`
* `epsilonSchedule`: `1.0 → 0.05` with `0.995` decay
* `targetUpdateInterval`: `500–4000` steps
* `trainInterval`: `1–4` steps
* `warmupSteps`: `1k–10k`

### B7: Stability Tricks (continued)

* **Reward scaling**: keep magnitudes in a small range to avoid exploding Q-values.
* **Gradient clipping**: prevent large weight updates.
* **Frame skipping**: skip some frames to reduce update frequency.
* **State stacking**: combine multiple frames to capture momentum.
* **Double DQN**: use separate networks for action selection and evaluation.
* **Huber loss**: more stable than MSE for outliers.

### B8: Evaluation Protocol

* Disable exploration: set `epsilon = 0`.
* Run multiple episodes (e.g., 20–50).
* Track mean, median, max, and standard deviation of scores.
* Optional: log action distribution to verify policy consistency.

---

## Common Pitfalls & Fixes

* **State is too minimal** → add missing features (velocity, distances).
* **Overfitting to specific patterns** → randomize pipes or gaps.
* **No learning** → check reward design and Q-value updates.
* **Unstable Q-values** → lower learning rate or add gradient clipping.

---

## Suggested Experiments

* Test different **reward designs**: only positive rewards, shaping bonuses, or heavy penalties.
* Try **Double DQN** for better stability.
* Explore **Dueling DQN** architecture.
* Use **Prioritized Experience Replay** to focus on important transitions.
* **Curriculum learning**: start with easy mode and gradually increase difficulty.

---

## Appendix: Math & Pseudocode

### Q‑Learning Update

```
q[s, a] ← q[s, a] + α * (r + γ * max(q[s', :]) − q[s, a])
```

### DQN Target

```
y = r + γ * (1 − done) * max(Q_target(s', a))
```

### Logging Helper

```python
def logEpisode(idx, reward, loss=None):
    print(f"Episode {idx}, Total Reward: {reward}, Loss: {loss}")
```

### Sanity Checks Before Training

* Ensure `reset()` sets all variables consistently.
* Verify that actions correspond to expected effects.
* Test `step()` manually to confirm state transitions and rewards.

---

**Next Steps:**

1. Start with tabular Q-learning → confirm learning pattern.
2. Replace Q-table with DQN neural network.
3. Experiment with hyperparameters and advanced techniques.