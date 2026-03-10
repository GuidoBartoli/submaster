"""Run the `submaster` CLI from the repository root helper script."""

from submaster.cli import main


if __name__ == "__main__":
    # Mirror the package entrypoint so local development can use `python main.py`.
    raise SystemExit(main())
