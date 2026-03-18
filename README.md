# Dual-Path Inference: GPU + Neural Engine on Apple Silicon

> **Archived.** This was the initial proof-of-concept (March 2026). The work evolved significantly and now lives in [orion-ane](https://github.com/MidasMulli/orion-ane):
> - **Dual inference engine** with smart routing (7 scenarios, 1.14x speedup): [`orion-ane/dual_inference/`](https://github.com/MidasMulli/orion-ane/tree/main/dual_inference)
> - **Memory daemon** with continuous enrichment (the real ANE value proposition): [`orion-ane/memory/`](https://github.com/MidasMulli/orion-ane/tree/main/memory)
> - **Key discovery:** Metal GPU is physically single-threaded — ANE is the only path to true parallelism on Apple Silicon.

---

Run two LLMs simultaneously on one chip — GPU cores for heavy inference, Neural Engine for lightweight tasks. No resource contention.

## Results

**Qwen 3.5 9B (GPU/MLX) + Llama 3.2 1B (ANE/CoreML) running concurrently on MacBook Air M5 (16GB):**

| Path | Solo | Concurrent | Delta |
|------|------|------------|-------|
| GPU (9B) | 23.4 tok/s | 19.9 tok/s | -15.0% |
| ANE (1B) | 53.8 tok/s | 45.7 tok/s | -0.4% to +1.3% |
| **Combined** | 77.2 tok/s | **65.6 tok/s** | |

The Neural Engine barely notices the GPU running. Combined throughput of 65.6 tok/s from two models at 6.6GB total memory.

**[View full interactive results →](results.html)** (open locally after cloning)

## What We Observed

1. **Independent silicon paths** — GPU and Neural Engine are separate compute blocks with near-zero mutual interference
2. **Dual-model inference works** — 9B + 1B coexist at ~6.6GB combined, well within 16GB
3. **ANE power efficiency** — 1B at 53.8 tok/s using ~2W vs 9B at 19.9 tok/s using ~15-20W (the tok/s difference is model size, not hardware speed — the power gap is the real insight)
4. **Model loading is the bottleneck** — ANE decode is 53.8 tok/s but subprocess init adds ~5s overhead; a persistent server would fix this

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   M5 AIR SoC                    │
│                                                 │
│  ┌──────────────┐  Unified  ┌────────────────┐  │
│  │  GPU CORES   │  Memory   │ NEURAL ENGINE  │  │
│  │  10 cores    │◄────────►│  16 cores       │  │
│  │  MLX runtime │  (shared) │  CoreML runtime │  │
│  │              │    bus    │                 │  │
│  │ Qwen3.5 9B  │          │ Llama3.2 1B    │  │
│  │ 4-bit (5.2GB)│          │ LUT6 (~1.4GB)  │  │
│  └──────────────┘          └────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Mac with Apple Silicon (M1-M5)
- [MLX](https://github.com/ml-explore/mlx) with `mlx-lm` installed
- [ANEMLL](https://github.com/Anemll/Anemll) with a converted ANE model
- Python 3.9+ with `aiohttp`

### 1. Start MLX server
```bash
python -m mlx_lm server \
  --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --port 8899 \
  --chat-template-args '{"enable_thinking":false}' \
  --prompt-cache-size 2
```

### 2. Set up ANEMLL model
```bash
# Clone and set up ANEMLL
git clone https://github.com/Anemll/Anemll.git anemll
cd anemll && ./create_uv_env.sh && source env-anemll/bin/activate

# Download pre-converted Llama 3.2 1B for ANE
git clone https://huggingface.co/anemll/anemll-meta-llama-Llama-3.2-1B-Instruct-ctx1024_0.3.5 models/llama-1b
```

### 3. Configure and run dashboard
Edit `server.py` to set your ANEMLL path, then:
```bash
pip install aiohttp
python server.py
# Dashboard at http://localhost:8430
```

### 4. Run tests
Open `http://localhost:8430`, select your ANE model, pick a prompt, hit **RUN TEST**.

The dashboard runs solo baselines for each path, then concurrent execution, and shows the throughput comparison with interference delta.

## ANE Models Tested

| Model | ANE tok/s | Size | Concurrent Delta | Verdict |
|-------|-----------|------|-----------------|---------|
| Qwen 2.5 0.5B | 72.6 | ~500MB | -6.3% | Fast but basic quality |
| **Llama 3.2 1B** | **53.8** | **~1.4GB** | **-0.4% to +1.3%** | **Sweet spot** |
| Qwen 2.5 3B | 13.0 | ~4.6GB | timeout | Too heavy for 16GB concurrent |

## Files

| File | Description |
|------|-------------|
| `server.py` | WebSocket server, inference orchestration, system stats |
| `dashboard.html` | Interactive dashboard with live test visualization |
| `results.html` | Static results page with all findings |
| `RESULTS.md` | Detailed writeup with analysis |

## Hardware

- MacBook Air M5, 16GB unified memory
- 10 GPU cores, 16 Neural Engine cores
- macOS Sequoia

## Credits

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for GPU inference
- [ANEMLL](https://github.com/Anemll/Anemll) — Neural Engine LLM inference
- Built during a [Claude Code](https://claude.ai/claude-code) session

## License

MIT
