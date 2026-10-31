import os
import shutil
import tempfile
import unittest

from ggufgen import build_gguf, kv_array_str, kv_f32, kv_string, kv_u32

from ggufpeek import read_gguf


class ReadGGUF(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "tiny.gguf")
        kvs = [
            kv_string("general.architecture", "llama"),
            kv_string("general.name", "tiny-test"),
            kv_u32("general.file_type", 15),
            kv_u32("llama.context_length", 2048),
            kv_u32("llama.block_count", 2),
            kv_u32("llama.attention.head_count", 8),
            kv_f32("llama.rope.freq_base", 10000.0),
            kv_array_str("tokenizer.ggml.tokens", ["a", "b", "c", "d"]),
        ]
        tensors = [
            ("token_embd.weight", [64, 100], 12),   # Q4_K
            ("output_norm.weight", [64], 0),         # F32
        ]
        build_gguf(self.path, kvs, tensors)
        self.g = read_gguf(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_header(self):
        self.assertEqual(self.g.version, 3)
        self.assertEqual(self.g.tensor_count, 2)
        self.assertEqual(self.g.architecture, "llama")
        self.assertEqual(self.g.name, "tiny-test")

    def test_arch_params(self):
        self.assertEqual(self.g.arch_get("context_length"), 2048)
        self.assertEqual(self.g.arch_get("block_count"), 2)
        self.assertEqual(self.g.arch_get("attention.head_count"), 8)
        self.assertAlmostEqual(self.g.arch_get("rope.freq_base"), 10000.0, places=2)
        self.assertIsNone(self.g.arch_get("nonexistent"))

    def test_tensors(self):
        embd = self.g.tensors[0]
        self.assertEqual(embd.name, "token_embd.weight")
        self.assertEqual(embd.dims, [64, 100])
        self.assertEqual(embd.type_name, "Q4_K")
        self.assertEqual(embd.n_params, 6400)
        self.assertEqual(self.g.tensors[1].type_name, "F32")
        self.assertEqual(self.g.total_params, 6400 + 64)

    def test_array_sampled(self):
        toks = self.g.metadata["tokenizer.ggml.tokens"]
        self.assertTrue(toks.get("__array__"))
        self.assertEqual(toks["len"], 4)
        self.assertEqual(toks["sample"], ["a", "b", "c", "d"])
        self.assertFalse(toks["truncated"])

    def test_large_array_truncates_but_reads_past(self):
        # A 50-element array must be fully consumed so the tensor after it still parses.
        path = os.path.join(self.tmp, "big.gguf")
        big = ["t%d" % i for i in range(50)]
        build_gguf(path, [kv_array_str("tokenizer.ggml.tokens", big)], [("w", [4, 4], 0)])
        g = read_gguf(path)
        arr = g.metadata["tokenizer.ggml.tokens"]
        self.assertEqual(arr["len"], 50)
        self.assertTrue(arr["truncated"])
        self.assertEqual(len(arr["sample"]), 16)
        self.assertEqual(g.tensor_count, 1)
        self.assertEqual(g.tensors[0].n_params, 16)

    def test_alignment_default(self):
        self.assertEqual(self.g.alignment, 32)


if __name__ == "__main__":
    unittest.main()
