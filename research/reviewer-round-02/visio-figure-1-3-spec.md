# Visio Figure 1–3 Revision Spec

## Fig. 1
- Restructure the figure into six blocks: feasible morphology generation, analytic response generation, DNN surrogate, descriptor-space optimization, feasible morphology projection, and physics-based stress testing.
- Replace any direct EnergyPlus/Radiance-to-2000-sample implication with `analytic response generator`.
- Replace the reward formula block with:
  z_i = (y_i - y_i^{min}) / (y_i^{max} - y_i^{min})
  u = (0,1,1)
  d_w = || w ⊙ (z-u) ||_2 / ||w||_2
  R = 1 - d_w
- Label `w` as `axis-scaling coefficients`.
- Current OCR extract still contains: `street loactions, Acotr netword, Critic netword`.

## Fig. 2
- Reframe the title and flow as `serialized static black-box search`.
- State explicitly: one episode = 40 sequential surrogate queries, reset = random action at episode start, termination = fixed 40-step horizon, and no physical time evolution.
- Action must be drawn as a 12-dimensional absolute normalized descriptor vector, not as an incremental perturbation of the previous morphology.
- Existing text extract availability: missing; use the task prompt as the authoritative rewrite spec.

## Fig. 3
- Replace network labels with: Actor network, Critic network, Target actor network, Target critic network.
- Define actor input/output, critic input, replay tuple, and soft update exactly as in the task brief.
- Add formula symbol definitions for Q, μ, γ, τ, N, θ, and θ′.
- Current OCR extract still contains legacy wording such as: `Target actor network`.