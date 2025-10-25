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
