# ggufpeek

**Inspect a GGUF model file without loading it.**

GGUF (the llama.cpp model format) puts everything you'd want to know about a
model — architecture, quantization, context length, tensor shapes — in a
self-describing header *before* the multi-gigabyte weights. `ggufpeek` reads only
that header and answers "what is this model?" in a few milliseconds, without
mapping a single byte of tensor data. Pure Python standard library — no
`gguf` package, no `torch`, no dependencies.

```bash
ggufpeek Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```
```
Meta-Llama-3-8B-Instruct.Q4_K_M.gguf  (4.58 GiB)
  GGUF version   3
  architecture   llama
  name           Meta-Llama-3-8B-Instruct
  parameters     8.03 B
  quantization   Q4_K  (file_type 15: Q4_K_M)
  alignment      32

  context length     8192
  embedding length   4096
  block count        32
  feed-forward       14336
  head count         32
  head count kv      8
  rope freq base     500000.0

  tensors        291
    Q4_K       161  ·  5.49 B
    Q6_K        65  ·  2.54 B
    F32         65  ·  266.24 K
```

---

## Why this exists

I keep a shelf of local models — different architectures, different
quantizations, confusingly similar filenames — and the question is always the
same: *what actually is this file?* What's its context length, is this the
Q4_K_M or the Q5, how many layers, is it even the architecture I think it is.
Loading it into llama.cpp to find out is slow and needs the whole model to fit;
`gguf`-the-Python-package pulls in NumPy for what is, underneath, a struct.

GGUF already carries all of that in a header a few kilobytes long. `ggufpeek`
just reads it.

## How it works

A GGUF file is a magic, a version, two counts, a flat **key/value metadata
block**, and a **tensor table** — all before the weights:

```
 GGUF │ ver │ n_tensors │ n_kv │ [ key, type, value ] × n_kv │ [ name, shape, type, offset ] × n_tensors │ …weights…
```

`ggufpeek` parses exactly that prefix with `struct` and stops at the weights.
From it, three things fall out:

- **The metadata**, read as typed key/values. Scalars are kept as-is; big arrays
  (the tokenizer vocab is 100k+ strings) are traversed but only sampled, so the
  reader stays light.
- **The tensor table** — every tensor's name, shape, ggml type, and offset — from
  which it sums parameters and builds a per-type histogram.
- **The effective quantization**, taken as the ggml type holding the most
  parameters rather than trusting a single label (see the notes).

## Install

```bash
pip install ggufpeek          # once published
# or, from a checkout:
pip install -e .
```

No dependencies. Python 3.9+.

## Use

**From the shell:**

```bash
ggufpeek model.gguf                 # the human summary above
ggufpeek model.gguf --tensors       # also list every tensor (type, shape, name)
ggufpeek model.gguf --json          # machine-readable summary
ggufpeek model.gguf --json --full   # …including all metadata and every tensor
```

**From Python:**

```python
from ggufpeek import read_gguf

g = read_gguf("model.gguf")
print(g.architecture, g.total_params)      # 'llama' 8030261248
print(g.arch_get("context_length"))        # 8192
for t in g.tensors[:3]:
    print(t.type_name, t.dims, t.name)      # 'Q4_K' [4096, 128256] 'token_embd.weight'
```

## Design notes & limitations

- **Header only, never the weights.** It reads the metadata and tensor table and
  stops. It will tell you a tensor's shape, type, and offset; it will not read or
  validate the quantized data itself.
- **GGUF v2/v3, little-endian.** That covers essentially every file in the wild.
  v1 (32-bit lengths) is rejected with a clear message rather than carried as a
  second code path; big-endian v3 files aren't handled.
- **Big arrays are sampled.** The tokenizer token/merge arrays are fully
  traversed (you can't skip a length-prefixed array without reading it) but only
  the first few elements are kept, with the true length recorded. `--json --full`
  still reports them this way, not as a 100k-line dump.
- **Quantization is inferred from the tensors.** `general.file_type` is shown, but
  the headline quant is the ggml type with the most parameters — see below.
- **`file_type` labels are partial.** The map covers the common ftypes; anything
  else shows as its raw number, never a wrong guess.

## Learning notes

*What I set out to learn, and what actually surprised me — this repo is partly a
notebook.*

- **The file already describes itself; reading labels beats loading.** The entire
  tool is small because GGUF front-loads every fact you want into a flat KV block.
  The insight worth keeping: "what is this model?" never actually needs the
  weights — the format was designed so a loader can plan before it allocates, and
  that same prefix is all a human needs too.
- **A length-prefixed array you don't want still has to be walked.** My first cut
  tried to seek past the tokenizer vocab to reach the tensor table faster — you
  can't: without reading each string's length you don't know where the next one
  starts. So I read every element but *keep* only a sample. Cheap to traverse,
  cheap to hold, and the tensor table after it still parses.
- **Trust the tensors over the label.** `general.file_type` is a single enum that
  has drifted across llama.cpp versions and is easy to mislabel. Summing
  parameters per ggml type and taking the majority gives the real quantization
  regardless of the enum — so I show the stated `file_type` but headline the
  histogram.
- **Round-trip the format to test it.** With no model files in the repo, I wrote a
  tiny GGUF *writer* for the tests and asserted the reader reads back exactly what
  the writer wrote. The serializer doubles as executable documentation of the
  layout — and caught two off-by-one width bugs immediately.
- **`struct` is the whole parser.** No framework, no schema library — a handful of
  `struct.unpack` calls against a format that is, underneath the ceremony, just
  little-endian integers and length-prefixed bytes. Naming the format honestly
  (it's a header, not a database) kept the code that size.

## Tests

```bash
pip install -e .
python -m unittest discover -s tests -v
```

The suite is fully offline and needs no model files: it writes minimal GGUF
files byte-for-byte and round-trips them through the reader — header, metadata,
typed arrays (including that a large array is sampled but fully consumed so the
tensor table after it still parses), the per-type histogram and JSON, and the
error paths (bad magic, truncation, unsupported version).

## License

MIT © Nirry
