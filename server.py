#!/usr/bin/env python3
"""
Dual-Path Inference Dashboard Server
Runs GPU (MLX) + ANE (ANEMLL) tests and streams results via WebSocket.
"""
import asyncio
import json
import os
import subprocess
import time
import re
import urllib.request
from pathlib import Path

from aiohttp import web

HOST = "0.0.0.0"
PORT = 8430
ROOT = Path(__file__).parent

MLX_BASE = "http://localhost:8899/v1/chat/completions"
MLX_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

ANEMLL_DIR = "/Users/midas/Desktop/cowork/anemll"
ANEMLL_ENV = os.path.join(ANEMLL_DIR, "env-anemll", "bin", "python")

ANE_MODELS = {
    "qwen-0.5b": {
        "name": "Qwen2.5-0.5B (ANE)",
        "chat": os.path.join(ANEMLL_DIR, "models", "qwen-0.5b", "chat.py"),
        "meta": os.path.join(ANEMLL_DIR, "models", "qwen-0.5b", "meta.yaml"),
        "params": "0.5B",
    },
    "llama-1b": {
        "name": "Llama3.2-1B-Instruct (ANE)",
        "chat": os.path.join(ANEMLL_DIR, "models", "llama-1b", "chat.py"),
        "meta": os.path.join(ANEMLL_DIR, "models", "llama-1b", "meta.yaml"),
        "params": "1B",
    },
    "qwen-3b": {
        "name": "Qwen2.5-3B (ANE)",
        "chat": os.path.join(ANEMLL_DIR, "models", "qwen-3b", "chat.py"),
        "meta": os.path.join(ANEMLL_DIR, "models", "qwen-3b", "meta.yaml"),
        "params": "3B",
    },
}

active_ane_model = "llama-1b"

TEST_PROMPTS = [
    "What is an ISDA Master Agreement and why does it matter for OTC derivatives trading?",
    "Explain the concept of netting in derivatives contracts.",
    "What is a Credit Support Annex and how does it relate to collateral management?",
]

ws_clients = set()


async def broadcast(msg):
    for ws in list(ws_clients):
        try:
            await ws.send_json(msg)
        except:
            ws_clients.discard(ws)


def run_mlx_inference(prompt, max_tokens=100):
    """Synchronous MLX inference via HTTP API."""
    start = time.time()
    payload = {
        "model": MLX_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        MLX_BASE,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        elapsed = time.time() - start
        return {
            "status": "ok",
            "model": MLX_MODEL,
            "time_s": round(elapsed, 2),
            "tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "tok_per_sec": round(usage.get("completion_tokens", 0) / elapsed, 1),
            "response": content[:500],
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "time_s": round(time.time() - start, 2)}


def run_ane_inference(prompt, max_tokens=100):
    """Synchronous ANE inference via ANEMLL subprocess."""
    global active_ane_model
    model_cfg = ANE_MODELS[active_ane_model]
    start = time.time()
    try:
        proc = subprocess.run(
            [ANEMLL_ENV, model_cfg["chat"], "--meta", model_cfg["meta"], "--prompt", prompt, "--max-tokens", str(max_tokens)],
            capture_output=True, text=True, timeout=300, cwd=ANEMLL_DIR,
        )
        elapsed = time.time() - start
        output = proc.stdout + "\n" + proc.stderr

        tok_s = 0
        total_tokens = 0
        prefill_tps = 0
        prefill_ms = 0
        total_time_s = 0
        for line in output.split("\n"):
            if "Inference:" in line and "t/s" in line:
                try:
                    tok_s = float(line.split("Inference:")[1].strip().split()[0])
                except:
                    pass
            if "Generated" in line and "tokens" in line:
                try:
                    total_tokens = int(line.split("Generated")[1].strip().split()[0])
                except:
                    pass
            if "Prefill:" in line and "t/s" in line:
                m = re.search(r'Prefill:\s*([\d.]+)ms\s*\(([\d.]+)\s*t/s\)', line)
                if m:
                    try:
                        prefill_ms = float(m.group(1))
                        prefill_tps = float(m.group(2))
                    except:
                        pass
            if "Total:" in line and "Generated" in line:
                m = re.search(r'in\s*([\d.]+)s', line)
                if m:
                    try:
                        total_time_s = float(m.group(1))
                    except:
                        pass
            if "t/s" in line and tok_s == 0:
                m = re.search(r'([\d.]+)\s*t/s', line)
                if m:
                    try:
                        tok_s = float(m.group(1))
                    except:
                        pass

        response = ""
        if "Assistant:" in output:
            response = output.split("Assistant:")[-1].strip()
            response = re.sub(r'\x1b\[[0-9;]*m', '', response)
            for marker in ["Prefill:", "\nPrefill", "\n\n"]:
                if marker in response:
                    response = response[:response.index(marker)]
            response = response[:500].strip()

        # Calculate model load time (elapsed minus actual inference time)
        load_time = round(elapsed - total_time_s, 2) if total_time_s > 0 else 0

        return {
            "status": "ok",
            "model": model_cfg["name"],
            "time_s": round(elapsed, 2),
            "tokens": total_tokens,
            "tok_per_sec": tok_s,
            "prefill_tps": prefill_tps,
            "prefill_ms": round(prefill_ms, 1),
            "decode_tps": tok_s,
            "load_time_s": load_time,
            "inference_time_s": round(total_time_s, 2),
            "response": response,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "time_s": round(time.time() - start, 2)}


def get_system_stats():
    """Get CPU, memory, GPU stats."""
    stats = {}
    try:
        top = subprocess.run(["top", "-l", "1", "-s", "0"], capture_output=True, text=True, timeout=5)
        for line in top.stdout.split("\n"):
            if "CPU usage" in line:
                m = re.search(r'([\d.]+)% user', line)
                if m:
                    stats["cpu_user"] = float(m.group(1))
                m = re.search(r'([\d.]+)% sys', line)
                if m:
                    stats["cpu_sys"] = float(m.group(1))
            if "PhysMem" in line:
                m = re.search(r'([\d.]+\w) used', line)
                if m:
                    stats["mem_used"] = m.group(1)
    except:
        pass

    try:
        mp = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=5)
        for line in mp.stdout.split("\n"):
            if "System-wide memory free percentage" in line:
                m = re.search(r'(\d+)%', line)
                if m:
                    stats["mem_free_pct"] = int(m.group(1))
    except:
        pass

    try:
        ioreg = subprocess.run(
            ["ioreg", "-r", "-n", "AppleARMIODevice", "-w", "0"],
            capture_output=True, text=True, timeout=5,
        )
        for line in ioreg.stdout.split("\n"):
            if '"die-temperature"' in line.lower() or '"temperature"' in line.lower():
                m = re.search(r'= (\d+)', line)
                if m:
                    stats["gpu_temp"] = int(m.group(1)) // 100
                    break
    except:
        pass

    return stats


async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws_clients.add(ws)
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                cmd = data.get("cmd")

                if cmd == "run_test":
                    prompt_idx = data.get("prompt_idx", 0)
                    prompt = TEST_PROMPTS[prompt_idx % len(TEST_PROMPTS)]
                    asyncio.create_task(run_full_test(prompt))

                elif cmd == "run_multi":
                    asyncio.create_task(run_multi_prompt_test())

                elif cmd == "set_ane_model":
                    global active_ane_model
                    model_key = data.get("model", "qwen-0.5b")
                    if model_key in ANE_MODELS:
                        active_ane_model = model_key
                        await broadcast({"type": "ane_model_changed", "model": model_key, "info": ANE_MODELS[model_key]})

    except Exception as e:
        print(f"WS error: {e}")
    finally:
        ws_clients.discard(ws)
    return ws


async def run_full_test(prompt):
    """Run solo baselines then concurrent test, streaming updates."""
    loop = asyncio.get_event_loop()

    await broadcast({"type": "test_start", "prompt": prompt})

    # Phase 1: Solo baselines
    await broadcast({"type": "phase", "phase": "solo", "label": "SOLO BASELINES"})

    # MLX solo
    await broadcast({"type": "path_start", "path": "mlx", "mode": "solo"})
    mlx_solo = await loop.run_in_executor(None, run_mlx_inference, prompt)
    await broadcast({"type": "path_result", "path": "mlx", "mode": "solo", "data": mlx_solo})

    # ANE solo
    await broadcast({"type": "path_start", "path": "ane", "mode": "solo"})
    ane_solo = await loop.run_in_executor(None, run_ane_inference, prompt)
    await broadcast({"type": "path_result", "path": "ane", "mode": "solo", "data": ane_solo})

    # Phase 2: Concurrent
    await broadcast({"type": "phase", "phase": "concurrent", "label": "CONCURRENT EXECUTION"})
    await broadcast({"type": "path_start", "path": "both", "mode": "concurrent"})

    # Run both in parallel
    mlx_future = loop.run_in_executor(None, run_mlx_inference, prompt)
    ane_future = loop.run_in_executor(None, run_ane_inference, prompt)

    mlx_conc, ane_conc = await asyncio.gather(mlx_future, ane_future)

    await broadcast({"type": "path_result", "path": "mlx", "mode": "concurrent", "data": mlx_conc})
    await broadcast({"type": "path_result", "path": "ane", "mode": "concurrent", "data": ane_conc})

    # Comparison
    comparison = {
        "mlx_solo_tps": mlx_solo.get("tok_per_sec", 0),
        "mlx_conc_tps": mlx_conc.get("tok_per_sec", 0),
        "ane_solo_tps": ane_solo.get("tok_per_sec", 0),
        "ane_conc_tps": ane_conc.get("tok_per_sec", 0),
    }
    if comparison["mlx_solo_tps"] > 0:
        comparison["mlx_delta"] = round(((comparison["mlx_conc_tps"] - comparison["mlx_solo_tps"]) / comparison["mlx_solo_tps"]) * 100, 1)
    if comparison["ane_solo_tps"] > 0:
        comparison["ane_delta"] = round(((comparison["ane_conc_tps"] - comparison["ane_solo_tps"]) / comparison["ane_solo_tps"]) * 100, 1)

    await broadcast({"type": "comparison", "data": comparison})
    await broadcast({"type": "test_complete"})


async def run_multi_prompt_test():
    """Run all test prompts sequentially for comprehensive results."""
    all_results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        await broadcast({"type": "multi_progress", "current": i + 1, "total": len(TEST_PROMPTS), "prompt": prompt})
        loop = asyncio.get_event_loop()

        mlx_solo = await loop.run_in_executor(None, run_mlx_inference, prompt)
        ane_solo = await loop.run_in_executor(None, run_ane_inference, prompt)

        mlx_future = loop.run_in_executor(None, run_mlx_inference, prompt)
        ane_future = loop.run_in_executor(None, run_ane_inference, prompt)
        mlx_conc, ane_conc = await asyncio.gather(mlx_future, ane_future)

        all_results.append({
            "prompt": prompt[:80],
            "mlx_solo": mlx_solo.get("tok_per_sec", 0),
            "mlx_conc": mlx_conc.get("tok_per_sec", 0),
            "ane_solo": ane_solo.get("tok_per_sec", 0),
            "ane_conc": ane_conc.get("tok_per_sec", 0),
        })
        await broadcast({"type": "multi_result", "index": i, "data": all_results[-1]})

    await broadcast({"type": "multi_complete", "results": all_results})


async def handle_stats(request):
    return web.json_response(get_system_stats())


async def handle_ane_models(request):
    """Return available ANE models and which are ready (meta.yaml exists)."""
    result = {}
    for key, cfg in ANE_MODELS.items():
        result[key] = {
            "name": cfg["name"],
            "params": cfg["params"],
            "ready": os.path.exists(cfg["meta"]),
        }
    return web.json_response({"models": result, "active": active_ane_model})


async def handle_index(request):
    return web.FileResponse(ROOT / "dashboard.html")


async def handle_results(request):
    return web.FileResponse(ROOT / "results.html")


app = web.Application()
app.router.add_get("/", handle_index)
app.router.add_get("/results", handle_results)
app.router.add_get("/ws", handle_ws)
app.router.add_get("/api/stats", handle_stats)
app.router.add_get("/api/ane-models", handle_ane_models)

if __name__ == "__main__":
    print(f"\n  DUAL-PATH INFERENCE DASHBOARD")
    print(f"  http://localhost:{PORT}\n")
    web.run_app(app, host=HOST, port=PORT, print=None)
