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

## Layout

The canvas uses a fixed TikZ bounding box of `17.5 cm x 6.6 cm`. The left side contains the replay batch notation, online actor, online critic, target actor, and target critic. The right side contains a single compact equation area with the TD target, critic loss, actor objective, and soft target updates.

The design uses solid dark arrows for forward tensor/data flow and dashed dark arrows for soft target-network updates. Gradient-update arrows and learning-rate values are intentionally omitted.

## Claim Boundary

The figure documents the implemented DDPG training architecture; it does not imply that the underlying urban-design task is a natural physical-time control process.
