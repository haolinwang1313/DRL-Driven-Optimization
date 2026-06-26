# Fig. 3 Round3 Design Note

## Role
Fig. 3 is now a DDPG learning architecture diagram for the surrogate-assisted optimization. It documents interaction/storage on the left and the online, target, and update components on the right.

## Layout
- Left side: `Environment` and `Experience replay buffer`.
- Upper right: online and target actor blocks.
- Middle right: online and target critic blocks.
- Lower right: TD target, critic loss, actor update, and a compact soft-update box.

## Boundary
This figure does not show start/end nodes, decision diamonds, query horizon logic, all-episode completion logic, or the single-step surrogate-query explanation. Those workflow details belong to Fig. 2.
