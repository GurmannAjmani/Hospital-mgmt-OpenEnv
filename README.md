---
title: Hospital Mgmt Env
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - rl
  - triage
---

# ER-Sentinel: Hospital Triage Management Environment

## Environment Description & Motivation
ER-Sentinel is a reinforcement learning and LLM evaluation environment that simulates the intense decision-making required in a busy hospital emergency room. The environment **trains a DQN (Deep Q-Network) agent** to act as the primary triage coordinator, tasked with allocating limited critical resources (ICU and Ward beds) to a queue of incoming patients whose health decays over time. 

The motivation is to train and evaluate decision-making agents using both **DQN-based reinforcement learning** and Large Language Models on pressure-testing, constraint-solving, and outcome optimization. The primary objective is to maximize hospital efficiency while minimizing the number of patient deaths. An episode fails instantly if deaths exceed the predefined collapse threshold. The DQN agent learns to make optimal triage decisions through reinforcement learning, balancing immediate needs against long-term hospital stability.

## Action and Observation Space

### Action Space (Discrete)
At each step, the agent must choose exactly **one** integer representing the triage decision for the patient currently at the front of the queue:
- `0`: **Wait** (Do nothing, leave patient in queue)
- `1`: **Admit to ICU** (Best for critical patients, severity 1-2)
- `2`: **Admit to Ward** (Best for mild patients, severity 3-5)
- `3`: **Discharge** (Release patient)

### Observation Space
The environment provides a comprehensive numerical and categorical view of the hospital state at every step:
- `er_queue_size` (int): Number of patients waiting in the emergency queue.
- `icu_available` (int): Number of currently open ICU beds.
- `ward_available` (int): Number of currently open Ward beds.
- `avg_patient_health` (float): Mean health of all waiting patients.
- `critical_count` (int): Number of waiting patients explicitly identified as critical (severity <= 2).
- `next_patient_severity` (int): Triage level of the front patient (1=Critical, 5=Mild).
- `next_patient_health` (float): Current HP of the front patient (out of 100.0).
- `deaths` (int): Accumulating death toll for the duration of the episode.
- `steps_remaining` (int): Steps until the shift successfully ends.
- `most_urgent_health` (float): Health of the patient with the lowest HP in the queue.

## Core Environment API

The environment follows the **OpenEnv interface**, enabling seamless interaction between LLMs and RL agents.

- **`reset(config: dict = None) -> HospitalMgmtObservation`**:
    - Initializes a new episode with a unique ID and `step_count = 0`.
    - Accepts an optional configuration to set `difficulty` ("easy", "medium", or "hard").
    - Pre-populates a queue of patients based on difficulty constraints.
    - Sets bed capacity, health decay rate, and the shift duration (`max_steps`).

- **`step(action: HospitalMgmtAction) -> HospitalMgmtObservation`**:
    - Increments the internal `step_count`.
    - Updates environment logic: Patient health decays based on their severity and environmental factors (Waiting patients lose HP every step).
    - Applies the selected action to the patient at the front of the queue.
    - Returns the updated observation, current reward, and the `done` status.

- **`state() -> State`**:
    - Returns the current environment metadata, specifically the `episode_id` and the current `step_count`.

## Reward Function and Termination

The environment employs a sophisticated dual-reward system designed to balance immediate precision with episodic outcomes. To ensure numerical stability and compatibility with diverse RL algorithms, **rewards are never exactly 0.0 and never exactly 1.0** (strictly within the range `(0.0, 1.0)`).

### 1. Dense Action Rewards
Assigned immediately after each `step()` based on the triage decision:
- **Optimal Outcome (0.95):** Perfect patient allocation (e.g., Critical → ICU). 
- **Suboptimal Outcome (0.80):** Functional but non-ideal allocation (e.g., Critical → Ward).
- **Inefficient Actions (0.30 - 0.40):** Mistake actions like attempting to admit to a full bed pool or waiting when beds are free for critical patients.
- **Failures (0.05):** The lowest possible reward, given for actions that actively harm the hospital state or violate basic logic.

### 2. Sparse Termination Rewards
Calculated only at the end of the episode to incentivize long-term patient survival:
- **Perfect Performance (0.45):** 0 deaths and all patients successfully treated.
- **Survivorship Bonus (0.20 - 0.35):** Base reward if no deaths occurred, varying by difficulty.
- **Death Penalty:** A linear penalty is subtracted for each death until the hospital collapses.
- **Collapse Floor (0.05):** The minimum sparse reward returned even in a total system failure.

### 3. Episode Termination (`done`)
An episode concludes immediately when either of these conditions is met:
- **Shift Duration:** The internal clock reaching `max_steps` (16 to 30 steps depending on difficulty).
- **Hospital Collapse:** The death toll reaching the difficulty threshold (4 for Easy, 2 for Medium/Hard). Once collapsed, the total reward for the episode is strictly limited.

## Task Descriptions & Difficulties

The environment supports three distinct difficulty configurations, directly impacting the constraint tightness, available beds, and patient deterioration rates:

| Task / Difficulty | Initial Patients | ICU Beds | Ward Beds | Decay Multiplier | Collapse Limit | Shift Duration (Steps) | Description |
|-------------------|------------------|----------|-----------|------------------|----------------|------------------------|-------------|
| **Easy** | 10 | 10 | 20 | Normal (1.0x) | 4 deaths | 30 steps | Plentiful resources and slower decay. Forgives heavy mistakes while agents grasp basic mechanics and ICU placements. |
| **Medium** | 12 | 4 | 8 | Fast (1.8x) | 2 deaths | 20 steps | Beds dynamically match patient severity. Agents must carefully distribute limited ICU vs Ward beds. |
| **Hard** | 12 | 3 | 9 | Extreme (2.4x) | 2 deaths | 16 steps | Severe shortage of ICU beds. Agents must prioritize effectively and make life-or-death capacity trades instantly. |

## Expected Baseline Scores

Performance is measured on a normalized scale [0.0–1.0], where 1.0 represents a perfect TRIAGE performance with zero deaths.

| Difficulty | Expected Reward (Normalized Score) |
|------------|------------------------------------|
| **Easy**   | 0.95+ |
| **Medium** | 0.80+ |
| **Hard**   | 0.40+ |

## Setup and Usage Instructions

### 0. Key Feature: DQN Agent Training
This environment enables **end-to-end Deep Q-Network (DQN) training** for autonomous hospital triage decision-making. The DQN agent learns optimal policies through reinforcement learning, adapting to different difficulty levels and resource constraints. The training framework includes:
- **Neural Network-based Q-learning** with experience replay and target networks
- **Multi-difficulty curriculum** (easy → medium → hard) for progressive learning
- **Real-time monitoring** via Streamlit dashboard to visualize training progress
- **Comparison with LLM baselines** for evaluating different decision-making approaches

### 1. Install Dependencies
Ensure you have the required packages installed in your Python environment:
```bash
pip install -r requirements.txt
```

### 2. Run the Inference Baseline (LLM Evaluation)
Before running the LLM evaluator, provide your API keys. Update the `.env` file in the directory to include your active `HF_TOKEN`.
```bash
python inference.py
```
*(No Docker required. The script safely wraps directly to the core python environment class and parses zero-shot actions automatically).*

### 3. Train a Custom DQN Agent (RL)
To train our native Reinforcement Learning agent from scratch:
```bash
python train.py --epochs 2000 --difficulty easy
```

### 4. Monitor via Streamlit Dashboard
Watch the DQN agent learn in real-time or monitor environment logs using the included visual dashboard:
```bash
streamlit run app.py
```
