"""Run the `submaster` CLI when the package is executed as a module."""

from submaster.cli import main


if __name__ == "__main__":
    # Delegate directly to the shared CLI entrypoint so `python -m submaster`
    # and the console script behave the same way.
    raise SystemExit(main())
