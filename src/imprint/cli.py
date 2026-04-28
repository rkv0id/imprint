"""Imprint command-line interface."""

import argparse
import sys

from imprint import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imprint",
        description="A Python library that gives AI agents memory.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"imprint {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
