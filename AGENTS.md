# AGENTS.md

## Project Guidelines

- Keep changes minimal and avoid overengineering.
- Remove temporary or large data files; never commit datasets.

## Conventions

- Python: PEP8 style, Ruff linting, NumPy-style docstrings.
- Use pytest for testing; prefer pandas/NumPy for data work.
- Use `uv run` for all Python execution (see `.cursor/rules/uv.mdc` if present).

## Agent Instructions

- **Specs and design docs:** Never commit design specs, implementation plans, or brainstorming write-ups. Write them only under the user Cursor folder outside this repo (`~/.cursor/superpowers/<repo-slug>/specs/` and `.../plans/`). If a spec must be shared, paste or attach it; do not commit specs.
- Run `uv run pytest -q` after changes and share results.
- Confirm before installing new dependencies.
- Never write secrets; always use environment variables.
- **`main` / default branch:** Do not merge into `main` or `git push origin main` unless the user explicitly asks (e.g. “push to main,” “ship to main”). Otherwise use a feature branch and push that branch only.

## Testing

- Use pytest for unit tests.
- Keep unit tests short, easy to read, and small in scope.
- Use descriptive names for each test.
