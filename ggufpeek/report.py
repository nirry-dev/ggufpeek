"""Turn a parsed GGUFFile into a human summary or a JSON document."""
from __future__ import annotations

import json
import os
from collections import Counter

from .gguf import GGUFFile

# A partial LLAMA_FTYPE map (general.file_type) for the common quantizations.
# The per-tensor type histogram is the authoritative signal; this is a friendly
# label for the ones we're confident about, and we fall back to the raw number.
FTYPE = {
    0: "all F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
}

# Architecture-scoped fields worth surfacing, in display order.
ARCH_FIELDS = [
    ("context length", "context_length"),
    ("embedding length", "embedding_length"),
    ("block count", "block_count"),
    ("feed-forward", "feed_forward_length"),
    ("head count", "attention.head_count"),
    ("head count kv", "attention.head_count_kv"),
    ("rope dim count", "rope.dimension_count"),
    ("rope freq base", "rope.freq_base"),
]


def human_count(n) -> str:
    n = float(n)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return str(int(n))


def human_bytes(n) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{int(x)} B" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} TiB"


def type_histograms(g: GGUFFile):
    """Return (count_by_type, params_by_type) over all tensors."""
    by_count: Counter = Counter()
    by_params: Counter = Counter()
    for t in g.tensors:
        by_count[t.type_name] += 1
        by_params[t.type_name] += t.n_params
    return by_count, by_params


def dominant_quant(g: GGUFFile):
    """The tensor type holding the most parameters — the model's effective quant."""
    _, by_params = type_histograms(g)
    if not by_params:
        return None
    return by_params.most_common(1)[0][0]


def render(g: GGUFFile) -> list:
    """A human-readable, multi-line summary."""
    lines = [f"{os.path.basename(g.path)}  ({human_bytes(g.size)})"]
    lines.append(f"  GGUF version   {g.version}")
    if g.architecture:
        lines.append(f"  architecture   {g.architecture}")
    if g.name:
        lines.append(f"  name           {g.name}")
    lines.append(f"  parameters     {human_count(g.total_params)}")

    dq = dominant_quant(g)
    ft = g.metadata.get("general.file_type")
    quant = f"  quantization   {dq or '—'}"
    if isinstance(ft, int):
        label = FTYPE.get(ft)
        quant += f"  (file_type {ft}{': ' + label if label else ''})"
    lines.append(quant)
    lines.append(f"  alignment      {g.alignment}")

    rows = [(label, g.arch_get(suffix)) for label, suffix in ARCH_FIELDS]
    rows = [(label, v) for label, v in rows if v is not None]
    if rows:
        lines.append("")
        for label, v in rows:
            lines.append(f"  {label:<18} {v}")

    by_count, by_params = type_histograms(g)
    lines.append("")
    lines.append(f"  tensors        {g.tensor_count}")
    for tname, cnt in by_count.most_common():
        lines.append(f"    {tname:<8} {cnt:>5}  ·  {human_count(by_params[tname])}")
    return lines


def to_json(g: GGUFFile, full: bool = False) -> str:
    by_count, by_params = type_histograms(g)
    doc = {
        "file": {
            "path": g.path,
            "name_on_disk": os.path.basename(g.path),
            "size": g.size,
            "gguf_version": g.version,
            "alignment": g.alignment,
        },
        "general": {
            "architecture": g.architecture,
            "name": g.name,
            "parameters": g.total_params,
            "file_type": g.metadata.get("general.file_type"),
            "dominant_quant": dominant_quant(g),
        },
        "tensor_count": g.tensor_count,
        "tensor_types": {
            name: {"tensors": by_count[name], "parameters": by_params[name]}
            for name, _ in by_count.most_common()
        },
    }
    if full:
        doc["metadata"] = g.metadata
        doc["tensors"] = [
            {
                "name": t.name,
                "dims": t.dims,
                "type": t.type_name,
                "offset": t.offset,
                "parameters": t.n_params,
            }
            for t in g.tensors
        ]
    return json.dumps(doc, indent=2, default=str)
