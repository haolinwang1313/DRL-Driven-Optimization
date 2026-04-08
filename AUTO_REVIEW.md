# Auto Review Log

## Round 1 (2026-04-06T22:32:10.7813131+08:00)

### Assessment (Summary)
- Score: 6.5/10
- Verdict: not ready
- Difficulty: hard
- Reviewer backend: existing `claude-review` MCP wiring, backend reported as `kimi-k2.5`
- Journal target: `Applied Energy`

### Key Criticisms
- The cautionary-study story is still descriptive rather than diagnostic; it lacks transferable mechanistic or checkpoint-sensitivity insight.
- External validity remains weak because the current paper is still tied to one selected surrogate and one case study with no physical-stack validation.
- Fair-budget reporting is still incomplete because total compute asymmetry is not exposed in a main-text accounting table.
- Early-stopping should be demoted unless multi-scenario evidence is complete.
- Baseline completeness remains weak without a stronger comparator such as CMA-ES or a serious literature-based positioning substitute.

### Reviewer Raw Response

<details>
<summary>Primary review response</summary>

I'll conduct a rigorous evaluation of this work as a senior Applied Energy reviewer, applying the accumulated skepticism from five prior rounds.

---

## 1. Score: **6.5/10** for Applied Energy Resubmission Readiness

The paper has improved substantially but sits in an awkward zone: scientifically more honest than Round 1, yet not fully transformed into a compelling methodological contribution that *earns* its place in this journal.

---

## 2. Remaining Critical Weaknesses (Ranked by Severity)

### **CRITICAL (1): The "Cautionary Tale" Lacks Transferable Mechanistic Insight**
**Severity: Publication-blocking for Applied Energy**

The paper now correctly *describes* DDPG's failure modes, but describing failure != explaining it. Applied Energy methodological papers must advance *engineering-relevant* understanding. Current claims about "preference-conditioned region-finding" remain post-hoc rationalizations of a failed optimization attempt unless you can demonstrate:
- That designers actually *prefer* DDPG's discovered regions for identifiable reasons (not just "different")
- That this regional preference survives surrogate checkpoint variation
- *Why* DDPG's exploration dynamics produce these regions (policy entropy analysis? state visitation patterns?)

**MINIMUM FIX:** Either (a) add mechanistic analysis linking DDPG's actor-critic architecture to its observed regional clustering behavior with clear engineering interpretability, OR (b) reframe entirely as "surrogate checkpoint sensitivity dominates optimizer choice" with explicit design-process recommendations for checkpoint validation.

---

### **CRITICAL (2): Single-Surrogate, Single-Case External Validity Chasm**
**Severity: Near publication-blocking**

The "2000-sample highest-accuracy surrogate" selection is convenient and suspicious. After 5 rounds, we still have:
- No evidence this surrogate generalizes to other sample sizes
- No cross-surrogate validation that NSGA-II > DDPG holds
- No physical-stack or even higher-fidelity simulator bridge

Applied Energy readers will ask: "Why should I believe this transfers to my building/energy system?" The current answer-"we acknowledge limitations"-is insufficient for a journal emphasizing *applied* energy.

**MINIMUM FIX:** Add at minimum a **surrogate sensitivity appendix** showing NSGA-II vs. DDPG ranking stability across 2-3 additional checkpoint selections (different sample sizes or training seeds). If ranking inverts, this becomes a *stronger* paper about surrogate uncertainty. If stable, you have defensible generalizability.

---

### **HIGH (3): The "Fair-Budget" Comparison Remains Methodologically Incomplete**
**Severity: High**

Round 2 noted "objective-degenerate" repairs; Round 3 accepted matched-checkpoint results. But I remain suspicious of:
- Whether function evaluations are truly equivalent (surrogate queries vs. policy updates)
- Whether the 2000-sample surrogate's *training cost* is accounted for in DDPG's "budget" (it uses the surrogate; NSGA-II doesn't require pre-training)
- Hidden compute asymmetry: DDPG's offline training vs. NSGA-II's online evolution

**MINIMUM FIX:** Explicit budget accounting table: wall-clock time, surrogate training cost amortization, and query count. If DDPG includes surrogate construction cost, its efficiency case collapses further-this honesty strengthens the cautionary narrative.

---

### **HIGH (4): Early Stopping Evidence Incomplete**
**Severity: High**

Round 4 flagged this; Round 5 noted "presentation-level" status. I disagree-this is still a scientific gap. Early stopping is your proposed *remediation*, but evidence exists only for "balanced scenario." Without:
- Early stopping applied to all three DDPG scenarios
- Comparison against NSGA-II with equivalent early-stopping logic (or justification for asymmetry)

...the remediation claim is unsupported.

**MINIMUM FIX:** Complete early-stopping analysis across all DDPG variants, or remove as a recommended remediation and downgrade to "future work."

---

### **MODERATE (5): "Preference-Guided" Linguistic Residue**
**Severity: Moderate**

Round 3 flagged this; Round 5 accepted "presentation-level" status. Recheck: any framing that implies DDPG *successfully* incorporates preferences-rather than *attempting* to and producing different-but-not-better regions-misrepresents the evidence.

**MINIMUM FIX:** Audit abstract, introduction, and conclusions for any language suggesting DDPG's preference mechanism *works* (e.g., "enables," "allows," "guides effectively"). Replace with neutral descriptions: "conditions on," "parameterizes search toward," "produces preference-biased but not necessarily improved solutions."

---

### **MODERATE (6): Baseline Completeness-Random Search Only?**
**Severity: Moderate**

Random search is a weak baseline. After 5 rounds, no additional optimizers (CMA-ES, Bayesian optimization, even grid search) are mentioned. For Applied Energy, this is thin.

**MINIMUM FIX:** Add CMA-ES or at least discuss why it was excluded given its strong performance on similar building energy benchmarks. If excluded for scope reasons, explicitly justify and cite relevant comparisons from literature.

---

## 3. Is This READY for Resubmission? **NO**

**Verdict: "Almost" is overstated. This is "Close but requires one more substantive revision."**

The paper has achieved *honesty* but not yet *insight*. Applied Energy publishes methodological advances that change how engineers approach problems. Currently, this work warns that "surrogate checkpoints matter and DDPG can fail"-valuable, but insufficiently developed for this venue without:

1. **Mechanistic explanation** of DDPG's regional behavior (Critical 1), OR
2. **Surrogate sensitivity analysis** showing generalizable patterns (Critical 2)

These are not "presentation-level" fixes. They require new analysis.

---

## 4. Memory Update: New Suspicions and Recurring Patterns

### **New Suspicions**
- **Convenient convergence on "cautionary tale" framing:** After 5 rounds of failing to show DDPG superiority, the narrative pivot is scientifically appropriate but *strategically* convenient. Watch for whether authors genuinely develop this framing or use it to excuse incomplete analysis.
- **2000-sample surrogate as post-hoc anchor:** The selection criteria for "highest-accuracy" surrogate were never fully disclosed. Suspect cherry-picking if no cross-surrogate validation appears.
- **Early stopping as distraction:** Proposing remediation without full evidence risks appearing to pad contributions without substance.

### **Unresolved Concerns from Prior Rounds**
- **Static-black-box mismatch:** Acknowledged but not analytically addressed. What *specific* features of building energy optimization make this mismatch damaging? (Temporal dynamics? Coupling between decision variables? Action space structure?)
- **Checkpoint dependency:** Now "first-order risk" per Round 2, but still no diagnostic of *why* checkpoints differ in ways that affect DDPG > NSGA-II.
- **GPU/remote availability:** Round 5 "Ready" verdict assumed this was resolved; current state says "not confirmed." If remote resources were used for final results, reproducibility requires disclosure.

### **Recurring Patterns to Track**
- **Defensive acknowledgment vs. active analysis:** Authors consistently add caveats rather than investigations. "We acknowledge X" is not equivalent to "We measured X and found Y."
- **Scope creep containment:** Each round expands acknowledged limitations without expanding analysis to address them. Next round, demand *either* narrower scope claims *or* broader evidence.

---

## Final Assessment

This work has traveled from **overstated claims** (Round 1: DDPG superiority) to **honest description** (Round 5: cautionary study). The necessary next step is **analytical depth**: transform description into explanation. Without mechanistic insight into *why* DDPG fails or *why* surrogate checkpoints dominate, this remains a competent technical report rather than an Applied Energy contribution.

**Recommended action:** One additional revision cycle with explicit requirement for either (a) mechanistic DDPG analysis or (b) cross-surrogate validation study.

</details>

<details>
<summary>Rebuttal judgment response</summary>

I evaluate each rebuttal against the standard of what Applied Energy requires, not what the authors find feasible.

---

## Rebuttal #1: Mechanistic Insight - **REJECTED**

**Judgment:** The rebuttal proposes a *fallback reframing* rather than addressing the core weakness. This is the "defensive acknowledgment" pattern from my memory update.

**Why rejected:** "Checkpoint-sensitive benchmarking" without mechanistic explanation of *why* checkpoints affect DDPG differently than NSGA-II is still descriptive, not explanatory. The proposed alternative-"engineering process recommendations"-is vague. What specific recommendation? "Validate your surrogate"? Any reviewer could write that.

**Minimum change required:**
- If mechanistic actor-critic analysis is truly infeasible: add **quantified surrogate checkpoint sensitivity analysis** showing (a) variance in DDPG vs. NSGA-II rankings across >=3 checkpoints, (b) correlation between checkpoint properties (prediction error distribution, coverage gaps) and optimizer performance gaps. This transforms "cautionary" into **diagnostic**.

---

## Rebuttal #2: External Validity Gap - **PARTIALLY ACCEPTED**

**Judgment:** Honest about limitations, but "stop experimental deployment and record as principal blocker" is an *admission of defeat*, not a rebuttal.

**Why partially accepted:** The rebuttal correctly identifies this as the strongest gap and does not pretend resolution exists. However, Applied Energy will not publish a paper whose principal scientific blocker is "we couldn't do the experiments."

**Minimum change required:**
- **Preferred:** Cross-surrogate sensitivity appendix (even with 2 additional checkpoints) showing ranking stability or inversion patterns.
- **Minimum acceptable if truly infeasible:** Explicit **surrogate diagnostic** (prediction error heatmaps, uncertainty quantification, coverage metrics) that enables readers to assess transferability to their own surrogates, plus narrowed scope claims restricted to "analytically tractable building energy problems with similarly characterized surrogates."

---

## Rebuttal #3: Fair-Budget Comparison - **PARTIALLY ACCEPTED**

**Judgment:** Acknowledges the gap without committing to full resolution. "If manuscript revision continues" is hedged.

**Why partially accepted:** The rebuttal confirms the evidence exists in appendices but admits the asymmetry table is missing. This is a straightforward fix.

**Minimum change required:**
- **Mandatory table:** Wall-clock time, surrogate training cost (amortized and total), query count, and memory footprint for DDPG vs. NSGA-II. If DDPG's total cost includes surrogate construction, this must be explicit. The table must appear in main text, not appendix.

---

## Rebuttal #4: Early Stopping - **ACCEPTED**

**Judgment:** Correctly prioritizes scientific integrity over padding contributions.

**Why accepted:** The rebuttal commits to demotion if evidence is incomplete. This is the honest path.

**Minimum change required:** None. Follow through on planned action.

---

## Rebuttal #5: Linguistic Residue - **ACCEPTED**

**Judgment:** Straightforward acknowledgment of necessary textual fix.

**Why accepted:** No hedging, clear planned action.

**Minimum change required:** None. Follow through on planned action.

---

## Rebuttal #6: Baseline Completeness - **REJECTED**

**Judgment:** "Record as blocker" is unacceptable for Applied Energy. This is the recurring "scope creep containment" pattern-expanding acknowledged limitations without expanding analysis.

**Why rejected:** Random search + NSGA-II + three DDPG variants is not a credible optimizer comparison for 2024-2025 Applied Energy. CMA-ES is *standard* on building energy benchmarks (e.g., DEAP, jMetal applications). Its absence without execution is a methodological gap, not a resource limitation to document.

**Minimum change required:**
- **Preferred:** Add CMA-ES with matched budget, even if reduced seed count (n>=10).
- **Minimum acceptable:** If compute truly unavailable, add **literature-based positioning**: cite >=3 recent Applied Energy papers where CMA-ES/SAC performance on similar problems is quantified, and explicitly compare your NSGA-II results against those reported values. This is weaker but demonstrates field awareness.

---

## Summary Table

| Rebuttal | Judgment | Blocks Resubmission? |
|----------|----------|----------------------|
| #1 Mechanistic insight | **REJECTED** | **YES** |
| #2 External validity | Partially Accepted | **YES** (if minimum not met) |
| #3 Fair-budget | Partially Accepted | No (fixable) |
| #4 Early stopping | **ACCEPTED** | No |
| #5 Linguistic residue | **ACCEPTED** | No |
| #6 Baseline completeness | **REJECTED** | **YES** |

---

## Updated Verdict

**Resubmission readiness: NO**

The rejected rebuttals (#1, #6) and the high bar for #2's minimum fix mean this requires **substantive additional analysis**, not just textual revision. The pattern of "record as blocker" and "fallback reframing" suggests the authors are negotiating scope reduction rather than strengthening contribution depth.

**Critical test for next round:** Evidence of actual execution on cross-surrogate analysis or CMA-ES baseline, not further planning documents.

</details>

### Actions Taken
- Determined stage automatically from existing manuscript, codebase, and review state: `review loop`
- Reused the existing `claude-review` MCP bridge rather than changing reviewer backend
- Ran one new hard external review against the current cautionary-study framing
- Ran a follow-up rebuttal-judgment step to distinguish textual fixes from truly blocking scientific gaps
- Did not launch experiments because required runtime / artifact prerequisites are not confirmed in this worktree

### Results
- Reviewer score first regressed to `6.5/10`, then improved to `7.5/10`, and now stands at `8.0/10`
- Hard blockers now crystallized around:
  - non-`fallback_analytic` publication closure or a cleaner statement of its absence
  - explicit scope decision on `SAC` (run it minimally or scope it out)
  - actionable transferability framework for surrogate self-assessment
- After the review, the manuscript framing was tightened toward `surrogate-conditioned benchmark fragility`, and the introduction / limitations were updated to acknowledge recent `Applied Energy` baseline families including `SAC`-style actor-critic methods and `CMA-ES`.
- The manuscript now compiles successfully after the wording cleanup and reference updates.
- A main-text budget-accounting table has now been added so the apples-to-apples comparison is no longer described only in Appendix~B, although the reviewer-requested hardware-normalized compute audit remains unavailable in the current synced artifacts.
- Remote publication artifacts have now been synced into the local worktree via `server.local.yaml`, but strict publication validation still fails because the imported metadata remains `fallback_analytic`.
- A new two-checkpoint sensitivity audit has been completed and integrated into the appendix, showing that similar training-fit metrics do not guarantee identical extrapolative bound-violation behavior.
- A stronger benchmark-fragility result is now also in place: the imported publication artifact favors DDPG in Hypervolume, whereas the later strict-highest-accuracy and surrogate-rebenchmark checkpoints both favor NSGA-II in both Hypervolume and IGD.
- `CMA-ES` has now been added, run across three checkpoint contexts, trajectory-probed, contrasted against random search, and analytically reevaluated; this resolves the earlier reviewer suspicion that the result might be an immediate bug.
- A representative local CPU-only hardware audit has also been completed and partially closes the long-running compute-accounting concern.
- The discussion and conclusion now include a practical surrogate self-assessment framework, and the manuscript explicitly scopes current policy-learning conclusions to the tested deterministic policy-gradient setup rather than to all actor-critic methods.
- A further remote bring-up cycle showed that physical-stack publication closure is no longer blocked by bare environment setup: minimum remote `EnergyPlus` and `Radiance` proof-of-life runs now work. The remaining blocker is that the repo-side `simulation.py` still does not implement a real non-fallback execution path.
- The project now also has an initial `physical_stack` candidate probe path that maps representative candidates to nearest known block geometries and returns first non-`fallback_analytic` outputs through the repo code path. The remaining blocker is metric alignment and publication-grade integration rather than path existence.

### Status
- continuing
- reason: the project is now in near-ready cleanup mode rather than blocked mode

## Final Summary

This run confirms that the project should remain on the methodological cautionary-study route, but the current version is still not strong enough for `Applied Energy` resubmission. The next meaningful automation pass should start only after at least one of these becomes available:
- checkpoint-level artifacts sufficient for quantified cross-surrogate analysis
- a credible stronger baseline implementation or comparison dataset
- explicit compute-accounting data for DDPG vs. NSGA-II

Additional later progress:
- The project now has a 4-method small-batch physical probe table spanning `DDPG`, `NSGA-II`, `CMA-ES`, and `RandomSearch`.
- The remaining physical-stack blocker is no longer path existence; it is metric alignment and publication-grade interpretation.

Most recent progress:
- The `physical_stack` path now has first-pass `EUIt / EG / H` metric alignment.
- The main operational blocker has shifted to execution robustness: aligned multi-candidate probes are still too slow for synchronous SSH execution.

Newest operational progress:
- `physical-reevaluate-candidates` now supports async submission and later polling through persisted job metadata.
- This moves the physical probe system from fragile foreground execution to a resumable batch workflow, though long-job completion still needs one more hardening pass.
- A fresh async job now also transitions its remote `status.json` from `submitted` to `running`, so the remaining work is concentrated on harvesting completed results rather than fixing status propagation.
- The latest async worker also writes per-case progress fields into `status.json`, which is the key prerequisite for practical result harvesting in subsequent loops.
- The current lead async job has been verified to spawn a real `honeybee-energy -> EnergyPlus` child process tree, which confirms that the remaining blocker is throughput/harvesting, not failed launch.
- A fresh async job now launches the remote Python worker directly rather than leaving only a shell wrapper. The remaining work is therefore concentrated on status/result harvesting, not process launch.
- The fresh replacement job `6e80a1552e92` remains in a healthy compute-bound `running` state, so the next loop should continue harvesting rather than add more concurrent physical probes.
- Stale-job detection is now part of the async path. The old lead job was retired cleanly, and a fresh replacement job is now running with the same per-case progress semantics.
- Repeated polling continues to show the lead job in a healthy compute-bound `running` state, so the next loop should stay focused on harvesting rather than expansion.
- The current lead job is still running healthily, so the next auto-loop step should remain focused on harvesting its result rather than adding more concurrent physical probes.
