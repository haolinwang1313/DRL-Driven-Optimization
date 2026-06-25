# Fig. 3 TikZ Design Note

This candidate replaces the legacy Visio-style Fig. 3 with a standalone TikZ rendering of the implemented DDPG actor-critic training architecture. It is a candidate artifact only: it does not overwrite `paper/manuscript/figures/fig3.pdf`, and it does not modify manuscript, appendix, response-letter, optimizer, or canonical-data files.

## Implementation Facts

- Source implementation: `paper_repro/optimizers.py`
- Runtime configuration: `configs/revision.yaml`
- State dimension: 3 target indicators (`EUIt`, `EG`, `H`)
- Action dimension: 12 normalized morphology descriptors
- Actor architecture: `3 -> 64 ReLU -> 32 ReLU -> 12 Sigmoid`
- Critic architecture: `15 -> 64 ReLU -> 32 ReLU -> 1`
- Replay mini-batch: `(s_i, a_i, r_i, s_{i+1}, d_i)`, `N = 128`
- Discount factor: `gamma = 0.999`
- Soft-update factor: `tau = 0.001`
- Actor learning rate: `0.001`
- Critic learning rate: `0.002`

## Layout

The canvas uses a fixed TikZ bounding box of `17.5 cm x 8.5 cm`. The upper band contains the replay mini-batch, online actor, online critic, target actor, target critic, and temporal-difference target. The lower band contains the arrow legend, actor objective, critic loss, and the separate actor/critic soft target updates.

The design uses orthogonal TikZ paths for all main arrows. Solid dark arrows indicate forward tensor/data flow, dashed rose arrows indicate gradient and optimizer updates, and dash-dot gray arrows indicate soft target-network updates.

## Claim Boundary

The figure documents the implemented DDPG training architecture; it does not imply that the underlying urban-design task is a natural physical-time control process.
