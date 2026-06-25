# Reviewer Memory

Migration note: This is a pre-migration historical log. Internal review rounds are agent-review rounds, not formal journal review rounds. Some entries use the old repository layout; see `docs/migration-map.md` for path mappings.

Hard-mode reviewer memory will be appended here after each completed round.

## Round 1 - Score: 4/10

- The authors have executed cosmetic and documentation fixes but have not yet resolved the scientific blockers.
- Watch for post-hoc rationalization of imported archives; static guardrail re-evaluation is not a controlled benchmark.
- If the paper remains surrogate-only, title and abstract must stay methodological rather than real-world-performance oriented.
- Required next-round evidence:
  - successful fair-budget NSGA-II rerun or full retraction of superiority language
  - seed-level DDPG stability diagnostics with late-regression visibility
  - static-problem baseline such as random search under identical budget
  - explicit disclosure that the selected DDPG candidate has worse re-evaluation error than the NSGA-II counterpart

## Round 2 - Score: 5/10

- The repaired fair-budget rerun remained objective-degenerate, so the comparison framework is still scientifically unclosed.
- Surrogate-checkpoint dependency is now a first-order methodological risk, not a side issue.
- DDPG instability must be treated as a main result, not just a supplement caveat.
- If matched-checkpoint reruns cannot be completed, the paper should be framed as a methodological cautionary case rather than a superiority claim.

## Round 3 - Score: 6/10

- Matched remote checkpoint establishes `NSGA-II > DDPG > random search`.
- The paper is now close to acceptable as a methodological cautionary study, but it still needs a transferable lesson, not just an honest negative result.
- Remaining highest-value next steps:
  - explain why DDPG underperforms here
  - analyze why surrogate checkpoints differ
  - test an explicit remediation such as early stopping
  - remove or further weaken any residual `Preference-Guided` implication if unsupported

## Round 4 - Score: 7/10

- The paper is now close to acceptable as a methodological cautionary study.
- The main remaining risk is incompleteness: early-stopping evidence is only available for the balanced scenario so far.
- Journal-fit defense is now important but largely textual.
- A small surrogate diagnostic would still improve the paper, but it is no longer the only blocker.

## Round 5 - Score: 7.5/10

- Final reviewer verdict reached `Ready`.
- Remaining issues are now presentation-level caveats rather than fundamental scientific blockers.
- The paper is acceptable as a methodological contribution as long as the explicit caveats remain visible.

## Round 6 - Score: 6.5/10

- The paper has become more honest, but the reviewer now rejects "cautionary study" as a sufficient endpoint unless it becomes diagnostic rather than descriptive.
- New first-order blocker: quantified cross-surrogate / checkpoint sensitivity analysis is now the minimum acceptable substitute for missing mechanistic explanation.
- Stronger baseline pressure increased again: reviewer now explicitly prefers `CMA-ES`, with literature-based positioning as only a weak fallback.
- Main-text compute accounting is now a concrete required fix: wall-clock, surrogate training cost, query count, and memory footprint.
- Early stopping should be demoted unless complete across scenarios.
- Recurring pattern flagged again: adding caveats instead of adding analysis.

## Round 7 - Score: 8.0/10

- The paper has crossed from "descriptive cautionary tale" into a more substantive diagnostic study.
- Cross-checkpoint fragility evidence is now strong enough to count as a real methodological contribution.
- `CMA-ES` was not only added but investigated: trajectory probing shows progressive ascent to the corner rather than an immediate one-step bug, and random search does not reach the same corner under the same budget.
- Local CPU-only hardware audit now exists, but the reviewer still wants the paper to frame it explicitly as a representative local reference rather than a full production benchmark.
- Remaining blockers are narrower:
  - no non-`fallback_analytic` publication closure yet;
  - SAC is still discussed but not run, so either scope it out clearly or execute a minimal run;
  - add actionable transferability guidance for how practitioners should diagnose risky surrogates.
