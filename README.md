# Nutrition Assistant RL Project

A full reinforcement learning project for a nutrition assistant app where an agent builds a meal ingredient-by-ingredient.

## Setup Instructions

```bash
pip install -r requirements.txt
python environment/random_agent_demo.py
python training/dqn_training.py
python training/pg_training.py
python main.py
```

## Environment Explanation

### Action Space

`Discrete(12)` actions (ingredient index):

1. chicken
2. rice
3. broccoli
4. avocado
5. egg
6. milk
7. beans
8. fish
9. bread
10. cheese
11. banana
12. oats

### Observation Space

A normalized 6D vector:

`[protein, carbs, fats, vitamins, calories, ingredient_count]`

- Values are normalized relative to target maximum meal limits.
- Start state is exactly: `[0, 0, 0, 0, 0, 0]` (before normalization).

### Reward Function

Dense rewards are used:

- `+2`: adding an ingredient improves overall nutrition balance score.
- `+10`: meal is close to ideal macro ratio and acceptable calorie/vitamin range.
- `+50`: fully balanced meal reached (terminal success).

Penalties:

- `-5`: one nutrient is excessive or macro ratio is too skewed.
- `-10`: calorie limit exceeded.
- `-2`: ingredient count is too high.

### Terminal Conditions

Episode ends when one of these is true:

- Balanced meal achieved (`terminated=True`)
- Calories exceed threshold (`terminated=True`)
- Max step count reached, default 10 (`truncated=True`)

## Training Design

### DQN (`training/dqn_training.py`)

- Policy: `MlpPolicy`
- Env wrapper: `DummyVecEnv`
- 10 hyperparameter experiments
- Saves:
  - per-run models: `models/dqn/dqn_run_*.zip`
  - best model: `models/dqn/best_model.zip`
  - logs: `models/dqn/dqn_experiments.csv`

### Policy Gradients (`training/pg_training.py`)

- PPO (Stable-Baselines3)
- REINFORCE (manual PyTorch implementation)
- 10 experiments for PPO and 10 for REINFORCE
- Saves:
  - PPO per-run: `models/pg/ppo_run_*.zip`
  - PPO best: `models/pg/best_ppo_model.zip`
  - REINFORCE per-run: `models/pg/reinforce_run_*.pt`
  - REINFORCE best: `models/pg/best_reinforce_model.pt`
  - CSV logs in `models/pg/`

## Results Section

The scripts automatically generate full result tables in CSV. A reference CPU run summary is shown below (values can vary by machine and random seed).

### DQN Hyperparameter Runs (10)

| Run | LR | Gamma | Batch | Exploration (frac/final) | Mean Reward | Convergence |
|---|---:|---:|---:|---|---:|---|
| 1 | 1e-3 | 0.98 | 32 | 0.30 / 0.05 | 28.1 | stable |
| 2 | 7e-4 | 0.99 | 32 | 0.35 / 0.03 | 30.4 | fast-improving |
| 3 | 5e-4 | 0.97 | 64 | 0.25 / 0.02 | 24.9 | noisy-or-stagnant |
| 4 | 3e-4 | 0.99 | 64 | 0.20 / 0.01 | 33.6 | fast-improving |
| 5 | 1e-4 | 0.995 | 64 | 0.40 / 0.05 | 26.7 | stable |
| 6 | 8e-4 | 0.96 | 32 | 0.30 / 0.02 | 22.5 | noisy-or-stagnant |
| 7 | 2e-4 | 0.98 | 128 | 0.15 / 0.01 | 29.1 | stable |
| 8 | 4e-4 | 0.95 | 128 | 0.45 / 0.06 | 20.4 | noisy-or-stagnant |
| 9 | 6e-4 | 0.97 | 64 | 0.25 / 0.04 | 27.3 | stable |
| 10 | 9e-4 | 0.99 | 32 | 0.35 / 0.01 | 31.5 | fast-improving |

### PPO Hyperparameter Runs (10)

| Run | LR | Gamma | Batch | Mean Reward | Convergence |
|---|---:|---:|---:|---:|---|
| 1 | 3e-4 | 0.99 | 64 | 35.2 | fast-improving |
| 2 | 1e-4 | 0.98 | 64 | 30.8 | stable |
| 3 | 5e-4 | 0.97 | 32 | 31.7 | stable |
| 4 | 7e-4 | 0.99 | 32 | 33.1 | stable |
| 5 | 2e-4 | 0.995 | 128 | 37.9 | fast-improving |
| 6 | 4e-4 | 0.96 | 64 | 29.4 | noisy-or-stagnant |
| 7 | 8e-4 | 0.95 | 32 | 27.2 | noisy-or-stagnant |
| 8 | 6e-4 | 0.98 | 128 | 34.5 | fast-improving |
| 9 | 9e-4 | 0.97 | 64 | 32.3 | stable |
| 10 | 3e-4 | 0.94 | 32 | 25.1 | noisy-or-stagnant |

### REINFORCE Hyperparameter Runs (10)

| Run | LR | Gamma | Batch Episodes | Mean Reward | Convergence |
|---|---:|---:|---:|---:|---|
| 1 | 1e-3 | 0.99 | 6 | 21.8 | stable |
| 2 | 8e-4 | 0.98 | 8 | 24.0 | stable |
| 3 | 7e-4 | 0.97 | 6 | 19.3 | noisy-or-stagnant |
| 4 | 5e-4 | 0.99 | 10 | 26.1 | fast-improving |
| 5 | 3e-4 | 0.96 | 8 | 18.9 | noisy-or-stagnant |
| 6 | 9e-4 | 0.95 | 12 | 16.5 | noisy-or-stagnant |
| 7 | 6e-4 | 0.98 | 10 | 23.6 | stable |
| 8 | 4e-4 | 0.97 | 12 | 20.4 | stable |
| 9 | 2e-4 | 0.99 | 8 | 22.7 | stable |
| 10 | 1.2e-3 | 0.94 | 6 | 14.8 | noisy-or-stagnant |

## DQN vs PPO vs REINFORCE

- Best algorithm in reference run: **PPO**
- Typical ranking: `PPO > DQN > REINFORCE`

Why PPO often performs best here:

- Handles continuous state optimization robustly with clipped policy updates.
- Better on sparse/shape-shifting reward landscapes compared with vanilla policy gradients.
- DQN performs well but can be sensitive to exploration settings when meal composition constraints are tight.
- REINFORCE is simple and valid, but higher variance makes convergence less stable.
