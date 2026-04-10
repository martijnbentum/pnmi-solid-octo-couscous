# Agent Notes

## Scope

This repository is a small Python library for computing PNMI and related
frame-level clustering metrics from phone labels and cluster labels.

## Code Style
- Match the existing style of the file you are editing, when in doubt use
  the style guides below as a reference.
- Use multi-line formatting for function definitions and function calls when
  they exceed the line limit.
- Target a maximum line length of 80 characters for new or modified lines when
  practical.
- Indent continued arguments by one level (4 spaces).
- Keep the closing parenthesis on the same line as the last argument.
- Do not vertically align arguments.
- Use single quotes unless the surrounding code clearly uses double quotes.
- Keep functions small and direct.
- Prefer explicit code over clever abstractions.
- Avoid unrelated refactors while making a requested change.
- Keep docstrings short and practical.
- For public or non-trivial functions, prefer the existing triple-single-quote
  docstring style used in this repo.
- Document parameters only when the signature or behavior is not obvious.
- When a parameter list is useful, keep it compact, for example:
  '''Short explanation.
  parameter_name: description
  '''
- Preserve the current public API unless the task explicitly calls for a change.


## API Conventions

- The package exposes its main public API from `pnmi/__init__.py` via explicit
  imports and `__all__`.
- Public helpers are plain functions rather than deep class hierarchies.
- Use `ValueError` for invalid caller input when the existing code already does
  so.

## Data And Dependencies

- NumPy is the main dependency and array handling should stay NumPy-first.
- Keep external dependencies minimal unless there is a strong reason to expand
  them.

## Tests

- Tests live in `tests/tester.py` and use `unittest`.
- Prefer small regression-style tests that exercise behavior directly.
- When adding functionality, add or update tests in the same style instead of
  introducing a different test framework.

## Change Discipline

- Favor backwards-compatible changes to the public API.
- Keep helper names descriptive and direct.
- If you touch packaging metadata, keep `pyproject.toml` minimal and consistent
  with the existing setuptools setup.
