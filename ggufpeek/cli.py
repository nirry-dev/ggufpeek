"""Command-line interface: ``ggufpeek <model.gguf>``."""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .gguf import GGUFError, read_gguf
from .report import render, to_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ggufpeek",
        description="Inspect a GGUF model file's header, metadata, and tensor table "
                    "without loading it.",
    )
    parser.add_argument("file", help="path to a .gguf model file")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    parser.add_argument(
        "--full", action="store_true",
        help="with --json, include all metadata and every tensor",
    )
    parser.add_argument(
        "--tensors", action="store_true",
        help="also list every tensor (type, shape, name)",
    )
    parser.add_argument("--version", action="version", version=f"ggufpeek {__version__}")

    args = parser.parse_args(argv)

    try:
        gguf = read_gguf(args.file)
    except (GGUFError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(to_json(gguf, full=args.full))
        return 0

    print("\n".join(render(gguf)))
    if args.tensors:
        print("\n  tensors:")
        for t in gguf.tensors:
            shape = "×".join(str(d) for d in t.dims) or "scalar"
            print(f"    {t.type_name:<8} {shape:<18} {t.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
