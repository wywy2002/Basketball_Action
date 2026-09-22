## Karpathy-Inspired Coding Guidelines

### 1. Think Before Coding

Do not assume. Do not hide confusion. Make tradeoffs explicit.

Before implementation:

- State your assumptions clearly. If you are uncertain, ask instead of guessing.
- When multiple interpretations are possible, present them rather than silently choosing one.
- If there is a simpler approach, say so. Push back when appropriate.
- If something is unclear, stop, identify the uncertainty, and ask for clarification.

### 2. Simplicity First

Write the minimum code needed to solve the requested problem. Avoid speculative work.

- Do not add features that were not requested.
- Do not create abstractions for code used only once.
- Do not add unrequested flexibility or configurability.
- Do not add error handling for scenarios that cannot realistically occur.
- If a 200-line solution could be a 50-line solution, simplify it.

Ask: would an experienced engineer consider this unnecessarily complex? If yes, simplify it.

### 3. Make Surgical Changes

Touch only what is necessary. Clean up only problems created by your own changes.

When modifying existing code:

- Do not “improve” adjacent code, comments, or formatting.
- Do not refactor code that is not broken or relevant to the request.
- Follow the existing style, even if you would normally choose another style.
- Mention unrelated dead code if noticed, but do not delete it.

When your changes create unused imports, variables, or functions, remove those. Do not remove pre-existing dead code unless explicitly asked.

Validation: every changed line should be directly traceable to the user’s request.

### 4. Execute Against Verifiable Goals

Define success criteria, then iterate until they are verified.

Turn instructions into measurable goals:

- “Add validation” → write invalid-input tests, then make them pass.
- “Fix the bug” → write a test that reproduces it, then make it pass.
- “Refactor X” → ensure tests pass both before and after the refactor.

For multi-step work, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria enable independent verification. Vague goals such as “make it work” do not.
