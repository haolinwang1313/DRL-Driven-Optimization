# Response Style Profile Guide

> Source adapted from the external `Substantive-Editing` writing workflow, then narrowed to this repository's revision and reviewer-response needs.

## Overview

Use this guide when drafting manuscript text, appendix text, reviewer-response language, revision trackers, or any other formal response-oriented prose in this repository.

The goal is not generic "good writing." The goal is consistent, evidence-bounded, engineering-style argumentation that matches the current paper objective:

- methodological caution over leaderboard framing;
- explicit dependence on surrogate reliability, benchmark fairness, and validation mode;
- clear boundaries around what the current evidence does and does not support.

## Core Style DNA

### Tone

Write in a restrained, technical, and reviewer-facing academic tone.

- Prefer `this study`, `this manuscript`, `this revision`, or `the present workflow` over strong first-person opinion.
- State what was done, what the evidence shows, and where the boundary remains.
- Treat limitations as part of the result, not as an afterthought.
- Keep the voice useful for engineering and methodological evaluation rather than promotional positioning.

### Sentence Pattern

Prefer medium-to-long sentences that explicitly carry:

1. context or condition;
2. the action or analysis performed;
3. the object being evaluated;
4. the bounded interpretation.

Typical paragraph motion:

1. state the scope or problem;
2. summarize what was added or checked;
3. report the evidence;
4. close with a narrow interpretation.

## Project Adaptation

### Repository-Specific Defaults

For this repository, the safest default is:

- do not write universal optimizer-superiority language;
- tie all comparative claims to the selected surrogate checkpoint or validation mode;
- make fairness, reevaluation drift, and checkpoint sensitivity explicit when relevant;
- do not imply physical-stack closure unless the evidence really comes from a non-fallback publication path.

### Default Framing Patterns

Use patterns like:

- `The evidence supports a methodological interpretation rather than a broad superiority claim.`
- `On the selected surrogate checkpoint, ...`
- `Under the current validation mode, ...`
- `This result should therefore be interpreted as ...`
- `The current evidence remains strongest for ..., weaker for ..., and still limited for ...`

Avoid patterns like:

- `substantially outperformed all baselines`
- `fully resolves`
- `proves`
- `general policy-learning superiority`
- `clear winner across settings`

## Required Workflow Before Writing

### 1. Read the Current Evidence

Before drafting, read the smallest set of files that define the current truth:

- `paper/manuscript/manuscript.tex`
- `paper/manuscript/appendix.tex`
- `paper/response/round-01/tracker/revision-tracker.md`
- `paper/response/round-01/tracker/revision-tracker.json`
- `paper/response/round-01/reviews/reviewer-memory.md`
- `paper/response/round-01/logs/paper-improvement-log.md`

Then read the task-specific artifacts that support the claim you are about to write.

### 2. Extract the Claim Boundary

Before drafting, write down internally:

- the exact claim;
- the direct evidence for that claim;
- the strongest limitation that must stay visible.

If any of those are missing, do not write confident prose yet.

### 3. Use a Revision Plan for Large Rewrites

If the user asks for a substantial rewrite of a section, first state:

- what will change;
- what evidence it will rely on;
- what claim will be weakened, strengthened, or removed.

### 4. Draft with Evidence-First Logic

Each important paragraph should follow a structure close to:

1. scope;
2. intervention or analysis;
3. evidence;
4. bounded takeaway.

### 5. Self-Audit Before Finalizing

Check the draft for:

- unsupported extrapolation;
- casual or promotional wording;
- vague references like `this result` without naming the result;
- missing condition phrases such as `on the selected surrogate checkpoint`;
- hidden contradictions with tracker state or manuscript conclusions.

## Preferred Writing Habits

### Good Habits

- Name the comparison basis explicitly.
- State whether evidence is surrogate-based, analytically reevaluated, or physically probed.
- Use layered contrast: `first`, `second`, `third`, or equivalent.
- Make reviewer-facing prose legible by separating action, evidence, and residual limit.
- Prefer concrete nouns and verbs over abstract rhetoric.

### Useful Verbs

Prefer:

- `clarify`
- `bound`
- `moderate`
- `reevaluate`
- `compare`
- `diagnose`
- `indicate`
- `support`
- `show`
- `remain`

Use stronger verbs such as `establish` or `demonstrate` only when the evidence really closes the point.

## Anti-Patterns

- Hype words without evidence.
- Broad claims that jump beyond the selected checkpoint, scenario, or validation mode.
- Reviewer responses that say only `fixed` without saying what changed.
- AI-sounding filler such as `delve`, `tapestry`, `unlock`, `game-changing`, or `holistic ecosystem`.
- Paragraphs that summarize conclusions before naming the evidence.
- Conclusions that introduce new claims not analyzed earlier in the paper.

## Reviewer-Response Pattern

For reviewer-facing updates, prefer this order:

1. identify the concern;
2. state the action taken;
3. point to concrete files, figures, tables, or artifacts;
4. state the remaining boundary if the point is only partially closed.

Example shape:

`We revised [target text or analysis] to address the concern regarding [issue]. Specifically, we added [change] and linked it to [evidence source]. The revised package now shows that [narrow supported conclusion]. This point is therefore [addressed / partially addressed] at the current revision stage, although [remaining limitation] still applies.`

## Reference Examples

See:

- `examples/skills/response-style-profile/reviewer-response-paragraph.md.template`
- `examples/skills/response-style-profile/manuscript-results-paragraph.md.template`

## Final Audit Checklist

- [ ] Does the paragraph name the evidence source?
- [ ] Does it state the comparison scope clearly?
- [ ] Does it avoid superiority language unless fully supported?
- [ ] Does it preserve the repo's cautionary-study framing?
- [ ] Does it leave the right limitation visible?
