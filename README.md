# A Comparative Study of Offline Reinforcement Learning Methods for Movie Recommendation

## Overview
- Offline recommendation on MovieLens-1M using Implicit Q-Learning (IQL) and Conservative Q-Learning (CQL).
- States combine static user features and recent watch sequences encoded with SASRec; policies act over the full movie catalog.
- NX_0 self-normalized importance sampling for offline evaluation alongside standard Top-K metrics.
- Demo notebook `demo_IQL&CQL.ipynb` walks through preprocessing and training end-to-end.
- Behavior Cloning (BC) warm-start available; CRR code stubbed for upcoming experiments; LinUCB bandit baseline planned.

## Algorithms & Status
- **IQL**: Main offline RL policy for ranking movies; uses expectile value regression and AWR policy updates.
- **CQL**: Conservative variant to reduce overestimation; parallel training/eval scripts mirror IQL.
- **BC**: Optional warm start for policy initialization.
- **CRR (TBC)**: Networks defined and a draft trainer script provided; more tuning to come.
- **LinUCB**: Contextual bandit baseline (disjoint + hybrid) using static user context and movie genres.

## Quickstart

### Offline RL (IQL/CQL)
```bash
pip install -r requirements.txt

# Preprocess MovieLens-1M (ensure datasets/{ratings,users,movies}.dat exist)
python scripts/preprocess_data.py --data_dir datasets --output_dir data/processed

# (Optional) Train BC warm start
python scripts/train_bc.py --data_dir data/processed --epochs 10

# Train IQL
python scripts/train_iql.py --data_dir data/processed --num_epochs 50

# Evaluate IQL with NX_0 + Top-K
python scripts/evaluate_iql.py \
  --data_dir data/processed \
  --iql_checkpoint checkpoints/iql/iql_best.pt \
  --bc_checkpoint checkpoints/bc/bc_policy_best.pt

# Train & evaluate CQL
python scripts/train_cql.py --data_dir data/processed --num_epochs 50
python scripts/evaluate_cql.py --data_dir data/processed --cql_checkpoint checkpoints/cql/cql_best.pt
```

### LinUCB Baseline (Replay on MovieLens-1M)
```bash
# From repo root; ensure datasets/{users,movies,ratings}.dat exist (MovieLens-1M)
pip install -r requirements.txt
python scripts/generate_ucb_replay_curves.py
```
- Uses random candidate set (k=20, includes the true movie) and up to 20K replay rounds.
- Outputs at repo root:
  - `UCB_replay.png`: combined reward/CTR curves and alpha sensitivity.
  - `UCB_replay1.png`: reward/CTR over time (Random vs Disjoint vs Hybrid LinUCB).
  - `UCB_replay2.png`: alpha sweep (CTR and cumulative reward) with bar summaries.

## Modeling Notes
- **State encoder**: Static demographics (gender, age bucket, occupation, zipcode bucket) + SASRec sequence encoder; concatenated embedding feeds policy/value networks.
- **Actions**: Predict over the full movie ID vocabulary; supports Top-K retrieval or direct action sampling.
- **Rewards**: Normalized movie ratings; configurable scaling for positive/negative balance.
- **IQL training**: Expectile value loss, TD Q-loss, and advantage-weighted regression with weight clipping; BC or SASRec weights can initialize the policy.
- **CQL training**: Adds conservative penalties to keep Q-values close to behavior support.
- **CRR (in progress)**: Advantage-weighted critic with future integration into the training loop.
**LinUCB**: Contextual bandit baseline (disjoint + hybrid) using static user context and movie genres.

## Evaluation
- NX_0 self-normalized importance sampling implemented in `src/evaluation/nx0_evaluator.py`.
- Top-K metrics (`recall@K`, `ndcg@K`, `hitrate@K`) in `src/evaluation/metrics.py`.
- Evaluation scripts accept BC checkpoints to estimate behavior policy needed for NX_0 weights.

**LinUCB evaluation**: Replay method with CTR and cumulative reward; random-k candidate set (k=20) or full-catalog.
