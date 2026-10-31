"""Serialize a minimal, valid GGUF file in memory — the mirror image of the
reader, used by the tests to round-trip the format."""
import struct


def gstr(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key, val):
    return gstr(key) + struct.pack("<I", 8) + gstr(val)


def kv_u32(key, val):
    return gstr(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def kv_f32(key, val):
    return gstr(key) + struct.pack("<I", 6) + struct.pack("<f", val)


def kv_bool(key, val):
    return gstr(key) + struct.pack("<I", 7) + struct.pack("<?", val)


def kv_array_str(key, vals):
    out = gstr(key) + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(vals))
    for v in vals:
        out += gstr(v)
    return out


def build_gguf(path, kvs, tensors, version=3):
    """Write a GGUF file. ``kvs`` is a list of serialized kv blobs (from the
    helpers above); ``tensors`` is a list of ``(name, dims, type_int)``."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(kvs)))
        for blob in kvs:
            f.write(blob)
        for name, dims, ttype in tensors:
            f.write(gstr(name))
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ttype))
            f.write(struct.pack("<Q", 0))  # data offset (no data section needed)
