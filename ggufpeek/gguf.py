"""A reader for the GGUF model-file format (llama.cpp), header and metadata only.

GGUF stores a model's weights after a self-describing header: a magic, a version,
counts, a flat key/value metadata block, and a tensor table (name, shape, type,
offset). ggufpeek parses exactly that prefix — a few KB to a few MB — and never
touches the multi-gigabyte tensor data that follows. That's the whole point:
answer "what is this model?" by reading its labels, not by loading it.

Layout (little-endian), GGUF v2/v3:

    magic  "GGUF"                     4 bytes
    version                           uint32
    tensor_count                      uint64
    metadata_kv_count                 uint64
    metadata_kv_count × {
        key        : string           uint64 len + bytes
        value_type : uint32
        value      : per type
    }
    tensor_count × {
        name         : string
        n_dimensions : uint32
        dimensions   : uint64 × n_dimensions
        type         : uint32          (a ggml tensor type)
        offset       : uint64
    }
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field


class GGUFError(Exception):
    """The file is not a GGUF we can read (bad magic, truncated, or unsupported)."""


# ggml tensor/quantization types (the `type` field of each tensor).
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0",
}

# GGUF metadata value types.
_STRING, _ARRAY = 8, 9
_SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
    12: ("<d", 8),
}

# Keep metadata light: huge arrays (tokenizer vocab, merges) are traversed but
# only this many elements are kept as a sample.
_ARRAY_SAMPLE = 16


@dataclass
class Tensor:
    name: str
    dims: list
    type: int
    type_name: str
    offset: int
    n_params: int


@dataclass
class GGUFFile:
    path: str
    size: int
    version: int
    metadata: dict
    tensors: list
    tensor_count: int

    @property
    def alignment(self) -> int:
        try:
            return int(self.metadata.get("general.alignment", 32))
        except (TypeError, ValueError):
            return 32

    @property
    def architecture(self):
        return self.metadata.get("general.architecture")

    @property
    def name(self):
        return self.metadata.get("general.name")

    @property
    def total_params(self) -> int:
        return sum(t.n_params for t in self.tensors)

    def arch_get(self, suffix, default=None):
        """Fetch an architecture-scoped key, e.g. ``llama.context_length``."""
        arch = self.architecture
        if not arch:
            return default
        return self.metadata.get(f"{arch}.{suffix}", default)


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) < n:
            raise GGUFError("unexpected end of file (truncated header)")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", "replace")

    def scalar(self, vtype: int):
        try:
            fmt, size = _SCALAR[vtype]
        except KeyError:
            raise GGUFError(f"unknown metadata value type {vtype}")
        return struct.unpack(fmt, self.read(size))[0]

    def value(self, vtype: int):
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            elem_type = self.u32()
            count = self.u64()
            return self._array(elem_type, count)
        return self.scalar(vtype)

    def _array(self, elem_type: int, count: int) -> dict:
        if elem_type == _ARRAY:
            raise GGUFError("nested arrays are not supported")
        sample = []
        for i in range(count):
            v = self.string() if elem_type == _STRING else self.scalar(elem_type)
            if i < _ARRAY_SAMPLE:
                sample.append(v)
        return {
            "__array__": True,
            "elem_type": elem_type,
            "len": count,
            "sample": sample,
            "truncated": count > _ARRAY_SAMPLE,
        }


def read_gguf(path: str) -> GGUFFile:
    """Parse the header, metadata, and tensor table of a GGUF file.

    Reads only the prefix — never the tensor data — so it's fast and cheap even
    on a multi-gigabyte model. Raises :class:`GGUFError` on a non-GGUF or
    truncated file, and ``OSError`` if the path can't be opened.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise GGUFError(f"not a GGUF file (bad magic {magic!r})")
        vbytes = f.read(4)
        if len(vbytes) < 4:
            raise GGUFError("truncated file (no version field)")
        version = struct.unpack("<I", vbytes)[0]
        if version < 2:
            raise GGUFError(f"unsupported GGUF version {version} (need v2 or newer)")

        r = _Reader(f)
        tensor_count = r.u64()
        kv_count = r.u64()

        metadata: dict = {}
        for _ in range(kv_count):
            key = r.string()
            vtype = r.u32()
            metadata[key] = r.value(vtype)

        tensors = []
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            n_params = 1
            for d in dims:
                n_params *= d
            tensors.append(
                Tensor(name, dims, ttype, GGML_TYPES.get(ttype, f"TYPE_{ttype}"), offset, n_params)
            )

    return GGUFFile(path, size, version, metadata, tensors, tensor_count)
