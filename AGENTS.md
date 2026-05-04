# Agent Instructions

These instructions apply to the whole repository.

## Required Checks

Before opening a PR or making a commit intended for a branch, make sure the same
checks used by GitHub Actions pass locally:

```bash
black --check .
pytest -q
sphinx-build docs docs/_build/html
```

If `black --check .` fails, run `black .`, review the diff, then rerun the
checks. If the local environment does not already have the development
dependencies installed, use:

```bash
python -m pip install -e '.[dev]'
```

## Commit Hygiene

- Do not commit generated cache or local machine artifacts such as
  `__pycache__/`, `.pytest_cache/`, `.coverage`, `.DS_Store`, or ad hoc lockfiles
  created by local tooling.
- Keep commits focused and include tests or docs for behavior changes.
- If a check cannot be run, state that clearly in the PR or handoff message with
  the reason.

## CI Notes

The relevant workflows are:

- `.github/workflows/run_tests.yml`
- `.github/workflows/build_docs.yml`

Both workflows run on `main` and `dev` pushes and pull requests. Documentation
deploys to GitHub Pages only from `main`.
