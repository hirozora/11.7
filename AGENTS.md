# Research-code conventions

- This repository is a research project, not a production service.
- Keep new experiment scripts direct, transparent, and reproducible.
- Do not add unrequested fallback paths, backward-compatibility layers, or
  artifact hash-matching logic.
- Prefer explicit inputs and immediate failures over automatic discovery or
  silent recovery.
