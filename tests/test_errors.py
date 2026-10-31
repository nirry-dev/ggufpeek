import os
import struct
import tempfile
import unittest

from ggufpeek import GGUFError, read_gguf


class Errors(unittest.TestCase):
    def _write(self, data: bytes) -> str:
        d = tempfile.mkdtemp()
        p = os.path.join(d, "x.gguf")
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_bad_magic(self):
        with self.assertRaises(GGUFError):
            read_gguf(self._write(b"NOPE" + b"\x00" * 16))

    def test_no_version(self):
        with self.assertRaises(GGUFError):
            read_gguf(self._write(b"GGUF"))

    def test_truncated_after_counts(self):
        # Valid magic + version, claims a tensor, then EOF.
        data = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 0)
        with self.assertRaises(GGUFError):
            read_gguf(self._write(data))

    def test_version_too_old(self):
        data = b"GGUF" + struct.pack("<I", 1) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
        with self.assertRaises(GGUFError):
            read_gguf(self._write(data))

    def test_missing_file(self):
        with self.assertRaises(OSError):
            read_gguf(os.path.join(tempfile.gettempdir(), "definitely-not-here-12345.gguf"))


if __name__ == "__main__":
    unittest.main()
