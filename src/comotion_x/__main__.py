"""Allow CoMotion-X to run with ``python -m comotion_x``."""

from comotion_x.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
