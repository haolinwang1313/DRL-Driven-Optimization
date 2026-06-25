# Fig. 2 TikZ Design Note

This candidate redraws Fig. 2 as a serialized static black-box search workflow for the DDPG benchmark. It is a candidate artifact only: it does not overwrite `paper/manuscript/figures/fig2.pdf`, and it does not modify manuscript, appendix, response-letter, optimizer, configuration, or canonical-result files.

## Implementation Facts

- Source implementation: `paper_repro/optimizers.py`
- Runtime configuration: `configs/revision.yaml`
- State dimension: 3 guarded target responses (`EUIt`, `EG`, `H`)
- Action dimension: 12 normalized morphology descriptors
- Action semantics: absolute descriptor action, not an incremental adjustment
- Actor output activation: Sigmoid
- Exploration: Gaussian noise with initial standard deviation `1.0`
- Action clipping: `clip(actor_action + noise, 0.0, 1.0)`
- Descriptor mapping: normalized action mapped to training-domain feature bounds
- Evaluator: selected DNN surrogate plus feature-distance guardrail, extrapolation penalty, and target-bound clipping
- Replay tuple: `(state, action, reward, next_state, done)`
- Episode length: 40 sequential surrogate queries
- Episodes per seed: 600
- Seeds per scenario: 20
- Replay capacity: `1,000,000`
- Replay batch size: 128
- Noise decay: `0.9998` per query/update step
- Scenario weights: Balanced `(1/3, 1/3, 1/3)`, Saving `(0.6, 0.2, 0.2)`, Generation `(0.2, 0.6, 0.2)`

## Layout

Panel a shows the per-step query loop: normalized state, policy query, exploration/clipping, absolute descriptor action, guarded surrogate evaluator, next state, reward, replay transition, replay buffer, and actor--critic update. The feedback loop routes the next normalized state back to the next query without implying physical time evolution.

Panel b shows the episode and training protocol: random reset, query 1, query 2, an omitted middle sequence, query 40, fixed-horizon terminal condition, and the three independently trained reward scenarios.

## Recommended Manuscript Text

The present environment serializes repeated queries to a static surrogate response surface. At the beginning of each episode, a random descriptor query is evaluated to construct the initial normalized three-target state. At each subsequent step, the actor outputs an absolute 12-dimensional normalized descriptor vector, rather than an incremental modification of the preceding morphology. The guarded surrogate maps this vector to the next target state and scalar reward. Accordingly, the 40-step horizon should be interpreted as a sequence of policy-training queries, not as physical time evolution.

## Claim Boundary

The sequence represents repeated policy-training queries to a static guarded surrogate and does not represent physical time evolution or incremental urban-morphology transformation.
