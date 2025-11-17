# Setup for batching Deepseek OCR using vLLM

Need vLLM nightly build.
```
uv venv
source .venv/bin/activate

uv pip install --pre --extra-index-url https://wheels.vllm.ai/nightly \
  "triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels" \
  vllm
```

If everything goes well,
```python3 batch_vllm_ocr.py```
should run Deepseek OCR batched over 32 pages using documents from `../data/sources` (can be configured using CLI args).

# Setup for synthetic metadata creation 

The `synth_metadata.py` file runs a small LM over an entire folder of markdown files,
scanned using a `corpus.jsonl` and creates a new `jsonl` file
with metadata of each markdown file, based on the required 
fields.

`build_prompt` contains the actual prompt, change the fields depending on what is required.

Crosscheck CONFIG at the beginning of the file.
```
MODEL_NAME = "google/gemma-3-4b-it"
CORPUS_PATH = "/scratch/akshit.kumar/md/5kcorpus.jsonl"
OUTPUT_META_PATH = "/scratch/akshit.kumar/md/5k_meta.jsonl"

BATCH_SIZE = 16
MAX_CHARS_MD = 64000  # 64k chars: enough for metadata
```

Run normally using `python3 synth_metadata.py`
