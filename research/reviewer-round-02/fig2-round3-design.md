# Fig. 2 Round3 Design Note

## Role
Fig. 2 is now a workflow flowchart for the DDPG-based surrogate search. It describes initialization, episode reset, repeated surrogate-query steps, termination decisions, and retained-candidate output.

## Layout
- Left side: one vertical, center-aligned workflow from `Start` to `End`.
- Right side: a dashed callout explaining one surrogate-query step.
- Main loop: `Query horizon reached?` returns to `Move to the next query step` on `No`.
- Episode loop: `All episodes / seeds completed?` returns to `Reset episode with a random descriptor query` on `No`.

## Boundary
This figure does not show target networks, TD targets, critic loss, actor-update formulas, replay-buffer architecture, or query-timeline panels. Those details belong to Fig. 3 or to the method text.
