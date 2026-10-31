import json
import os
import shutil
import tempfile
import unittest

from ggufgen import build_gguf, kv_string, kv_u32

from ggufpeek import read_gguf
from ggufpeek.report import dominant_quant, human_bytes, human_count, render, to_json


class Human(unittest.TestCase):
    def test_count(self):
        self.assertEqual(human_count(6400), "6.40 K")
        self.assertEqual(human_count(8_030_000_000), "8.03 B")
        self.assertEqual(human_count(1_200_000), "1.20 M")
        self.assertEqual(human_count(500), "500")

    def test_bytes(self):
        self.assertEqual(human_bytes(512), "512 B")
        self.assertEqual(human_bytes(1536), "1.50 KiB")
        self.assertTrue(human_bytes(4_380_000_000).endswith("GiB"))


class Summaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "m.gguf")
        build_gguf(
            self.path,
            [kv_string("general.architecture", "llama"), kv_u32("llama.context_length", 4096)],
            [("a.weight", [10, 10], 12), ("b.weight", [10, 10], 12), ("n.weight", [10], 0)],
        )
        self.g = read_gguf(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dominant_quant(self):
        # 200 params of Q4_K vs 10 of F32 -> Q4_K dominates.
        self.assertEqual(dominant_quant(self.g), "Q4_K")

    def test_render_has_fields(self):
        text = "\n".join(render(self.g))
        self.assertIn("architecture   llama", text)
        self.assertIn("context length", text)
        self.assertIn("Q4_K", text)

    def test_json(self):
        doc = json.loads(to_json(self.g))
        self.assertEqual(doc["general"]["architecture"], "llama")
        self.assertEqual(doc["general"]["dominant_quant"], "Q4_K")
        self.assertEqual(doc["tensor_count"], 3)
        self.assertEqual(doc["tensor_types"]["Q4_K"]["tensors"], 2)
        self.assertNotIn("tensors", doc)

    def test_json_full_includes_tensors(self):
        doc = json.loads(to_json(self.g, full=True))
        self.assertEqual(len(doc["tensors"]), 3)
        self.assertIn("metadata", doc)


if __name__ == "__main__":
    unittest.main()
