"""Peek at a GGUF model file from Python.

    python examples/peek.py path/to/model.gguf

Prints the human summary, then the model's parameter count and effective
quantization pulled straight from the API.
"""
import sys

from ggufpeek import read_gguf
from ggufpeek.report import dominant_quant, human_count, render

if len(sys.argv) != 2:
    sys.exit("usage: python examples/peek.py <model.gguf>")

gguf = read_gguf(sys.argv[1])
print("\n".join(render(gguf)))
print(f"\n-> {human_count(gguf.total_params)} parameters, dominant quant {dominant_quant(gguf)}")
