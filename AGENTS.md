# Agent Notes

## Scope

This repository is a small Python library for computing PNMI and related
frame-level clustering metrics from phone labels and cluster labels.

## Code Style
## Coding Style
- Target a maximum line length of 80 characters for new or modified lines when
  practical.
- Style priority: follow the rules in this file over nearby inconsistent code.
- Keep function calls and definitions on one line when practical; only wrap
  once the line length would exceed the limit.
- Indent continued arguments by one level (4 spaces).
- Keep the closing parenthesis on the same line as the last argument.
- Do not vertically align arguments.
- For wrapped function calls, keep as much as possible on the first line and
  continue on the next line with one extra indent level (4 spaces).
- Preferred wrapped call style:
  result = some_function(first_argument, second_argument,
      third_argument, fourth_argument)
- Avoid this style:
  result = some_function(
      first_argument,
      second_argument)
- Prefer the same wrapping style for `return` statements:
  return some_function(first_argument, second_argument,
      third_argument)
- For wrapped function definitions, use this style:
  def some_function(first_argument, second_argument, third_argument,
      fourth_argument=None):
- Do not add spaces around `=` in keyword defaults in function definitions.
  Use `fourth_argument=None`, not `fourth_argument = None`.
- Order imports as standard-library first, third-party second, and local
  imports last.
- Use blank lines between import groups.
- Order imports alphabetically within each import group.
- Use single quotes unless the surrounding code clearly uses double quotes.
- Keep functions small and direct.
- Prefer explicit code over clever abstractions.
- Use compact style for simple conditionals when they remain readable,
  for example `if x is None: x = default`.
- Avoid long conditionals. If a conditional is getting long, simplify the logic
  or split the function into smaller parts.
- For long `raise ValueError(...)` messages, build a variable first and raise
  with that variable instead of wrapping a long inline string.
- Use one-line docstrings for small internal helpers unless the behavior is not
  obvious.
- Keep docstrings short and practical. Document parameters for main functions.
- For main/public functions, document parameters when useful with the style
  below.
- When a parameter list is useful, keep it compact, for example:
  '''Short explanation.
  audio_filename:    path to the audio file
  start:             segment start time in seconds
  '''
- If a parameter list is present in a docstring, align the descriptions by
  padding after the colon based on the longest parameter name in that
  docstring block.
- Private/internal helpers should not have parameter lists unless the behavior
  is not obvious.
- Use the triple-single-quote docstring style.
- When coding style is clear from the rules above, change code to match the
  style instead of preserving inconsistent older formatting.
- Avoid wrapped imports.
- Prefer compact import lines such as `from x import a, b, c` up to the line
  width limit.
- If many names are imported from the same module, prefer importing the module
  instead to keep the origin clear.



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
