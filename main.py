"""Run the `submaster` CLI from the repository root helper script."""

from submaster.cli import main

# TODO:
# - Improve spinner performance and appearance.
# - Add documentation for all functions and classes in the codebase, including docstrings for parameters and return types.
# - Add type annotations for all functions and variables in the codebase.


if __name__ == "__main__":
    # Mirror the package entrypoint so local development can use `python main.py`.
    raise SystemExit(main())
