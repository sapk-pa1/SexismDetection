# Code comments: short and useful

- Do comment code, but keep each comment brief: one line, two at most.
- Comment the *why* and the intent of a block: a short note above each function's main steps, non-obvious choices, paper/metric references, library workarounds, and research rules being enforced.
- Don't comment every line, and don't restate what is obvious from the code (`# return result`, `i += 1  # increment i`).
- No long comment blocks, banners or section dividers; no commented-out code.
- Docstrings: one line for public functions; add `Args`/`Returns` only when a shape, unit or label order isn't clear from type hints.
- YAML configs: short comments on non-obvious values and `TODO:` markers.
- When editing existing code, don't add comments to lines you didn't change.
