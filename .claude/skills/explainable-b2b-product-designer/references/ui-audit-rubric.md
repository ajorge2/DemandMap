# UI Audit Rubric

Score each category 0 (blocked), 1 (workable), or 2 (clear). A strong release has no zero and at least 18/22.

1. **Audience:** The interface names a recognizable operator and their goal.
2. **Orientation:** A first-time user can see the workflow and current step.
3. **Terminology:** Primary labels describe user outcomes rather than implementation.
4. **Control consequences:** Every control explains what changes when it moves.
5. **Defaults:** Recommended defaults are visible and reversible.
6. **Primary action:** The next useful action is visually and verbally dominant.
7. **Results:** Conclusions appear before raw requests, diagnostics, or audit logs.
8. **Trust:** Generated interpretation is distinguishable from source evidence.
9. **Progressive disclosure:** Technical detail remains available without blocking the main path.
10. **System states:** Empty, loading, success, degraded, and error states provide a next action.
11. **Accessibility:** Keyboard, focus, labels, contrast, responsive layout, and motion preferences are covered.

## Five-second comprehension check

A new user should be able to answer:

- What does this product help me decide?
- What data is it using?
- What should I do next?
- What will that action change or send externally?
- Where can I inspect the evidence?

## Explainable-AI check

For every generated result, verify that the interface exposes:

- the input scope;
- the distinction between source evidence and interpretation;
- meaningful uncertainty or evidence strength;
- a way to inspect representative source data; and
- a recovery path when generation fails.
