#!/usr/bin/env python3
"""
Dual-Path Inference Test
Runs GPU (MLX) and ANE (ANEMLL) simultaneously on the same prompt.
Tests whether Neural Engine and GPU can operate concurrently on Apple Silicon.
"""
import subprocess
import threading
import time
import json
import urllib.request
import os

MLX_BASE = "http://localhost:8899/v1/chat/completions"
MLX_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

ANEMLL_DIR = "/Users/midas/Desktop/cowork/anemll"
ANEMLL_ENV = os.path.join(ANEMLL_DIR, "env-anemll", "bin", "python")
ANEMLL_CHAT = os.path.join(ANEMLL_DIR, "models", "qwen-0.5b", "chat.py")
ANEMLL_META = os.path.join(ANEMLL_DIR, "models", "qwen-0.5b", "meta.yaml")

TEST_PROMPT = "What is an ISDA Master Agreement and why does it matter for OTC derivatives trading?"

results = {}


def run_mlx_inference():
    """Path A: GPU via MLX"""
    start = time.time()
    payload = {
        "model": MLX_MODEL,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": 200,
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
        results["mlx"] = {
            "path": "GPU (MLX)",
            "model": MLX_MODEL,
            "time_s": round(elapsed, 2),
            "tokens_generated": usage.get("completion_tokens", "?"),
            "prompt_tokens": usage.get("prompt_tokens", "?"),
            "tok_per_sec": round(usage.get("completion_tokens", 0) / elapsed, 1) if elapsed > 0 else "?",
            "response_preview": content[:300],
        }
    except Exception as e:
        results["mlx"] = {"error": str(e)}


def run_ane_inference():
    """Path B: Neural Engine via ANEMLL"""
    start = time.time()
    try:
        proc = subprocess.run(
            [
                ANEMLL_ENV, ANEMLL_CHAT,
                "--meta", ANEMLL_META,
                "--prompt", TEST_PROMPT,
                "--max-tokens", "200",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=ANEMLL_DIR,
        )
        elapsed = time.time() - start
        # ANEMLL prints to both stdout and stderr — combine them
        output = proc.stdout + "\n" + proc.stderr

        # Parse tok/s from ANEMLL output
        tok_s = "?"
        total_tokens = "?"
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
            # Also try parsing the inline tok/s like "77.1 t/s"
            if "t/s" in line and tok_s == "?":
                import re
                m = re.search(r'([\d.]+)\s*t/s', line)
                if m:
                    try:
                        tok_s = float(m.group(1))
                    except:
                        pass

        # Extract just the assistant response
        response = ""
        if "Assistant:" in output:
            response = output.split("Assistant:")[-1].strip()
            # Remove ANSI codes
            import re
            response = re.sub(r'\x1b\[[0-9;]*m', '', response)
            # Trim at the stats line
            for marker in ["Prefill:", "\nPrefill", "\n\n"]:
                if marker in response:
                    response = response[:response.index(marker)]
            response = response[:300].strip()

        results["ane"] = {
            "path": "Neural Engine (ANEMLL/CoreML)",
            "model": "Qwen2.5-0.5B (ANE-optimized)",
            "time_s": round(elapsed, 2),
            "tokens_generated": total_tokens,
            "tok_per_sec": tok_s,
            "response_preview": response,
        }
        if proc.returncode != 0:
            results["ane"]["stderr"] = proc.stderr[:500]
    except Exception as e:
        results["ane"] = {"error": str(e)}


def main():
    print("=" * 70)
    print("  DUAL-PATH INFERENCE TEST")
    print("  GPU (MLX) + Neural Engine (ANEMLL) — Simultaneous Execution")
    print("  Apple M5 Air · 16GB Unified Memory")
    print("=" * 70)
    print(f"\nPrompt: {TEST_PROMPT}")
    print()

    # --- Solo baselines ---
    print("━" * 70)
    print("  PHASE 1: SOLO BASELINES")
    print("━" * 70)

    print("\n🔵 Running MLX (GPU) solo...")
    t_start = time.time()
    run_mlx_inference()
    mlx_solo_time = time.time() - t_start
    mlx_result = results.get("mlx", {})
    if "error" in mlx_result:
        print(f"   ❌ MLX Error: {mlx_result['error']}")
    else:
        print(f"   ✅ {mlx_result['tok_per_sec']} tok/s | {mlx_result['time_s']}s | {mlx_result['tokens_generated']} tokens")

    print("\n🟢 Running ANE (Neural Engine) solo...")
    results.clear()
    t_start = time.time()
    run_ane_inference()
    ane_solo_time = time.time() - t_start
    ane_result_solo = results.get("ane", {})
    if "error" in ane_result_solo:
        print(f"   ❌ ANE Error: {ane_result_solo['error']}")
    else:
        print(f"   ✅ {ane_result_solo['tok_per_sec']} tok/s | {ane_result_solo['time_s']}s | {ane_result_solo['tokens_generated']} tokens")

    # Store solo results
    solo_results = {
        "mlx_solo": mlx_result,
        "ane_solo": dict(ane_result_solo),
    }

    # --- Concurrent test ---
    print()
    print("━" * 70)
    print("  PHASE 2: CONCURRENT EXECUTION (THE BIG TEST)")
    print("━" * 70)
    print("\n🚀 Launching BOTH paths simultaneously...")
    results.clear()

    t_start = time.time()
    t1 = threading.Thread(target=run_mlx_inference, name="MLX-GPU")
    t2 = threading.Thread(target=run_ane_inference, name="ANE")

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    total_concurrent = time.time() - t_start

    # --- Report ---
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    for path_key, label in [("mlx", "GPU (MLX)"), ("ane", "Neural Engine (ANE)")]:
        data = results.get(path_key, {})
        print(f"\n📍 {label}:")
        if "error" in data:
            print(f"   ❌ Error: {data['error']}")
        else:
            print(f"   Model:     {data.get('model', '?')}")
            print(f"   Time:      {data.get('time_s', '?')}s")
            print(f"   Tokens:    {data.get('tokens_generated', '?')}")
            print(f"   Speed:     {data.get('tok_per_sec', '?')} tok/s")
            print(f"   Response:  {data.get('response_preview', '')[:150]}...")

    print()
    print("━" * 70)
    print("  COMPARISON")
    print("━" * 70)

    mlx_concurrent = results.get("mlx", {})
    ane_concurrent = results.get("ane", {})

    mlx_solo_speed = solo_results["mlx_solo"].get("tok_per_sec", 0)
    mlx_conc_speed = mlx_concurrent.get("tok_per_sec", 0)
    ane_solo_speed = solo_results["ane_solo"].get("tok_per_sec", 0)
    ane_conc_speed = ane_concurrent.get("tok_per_sec", 0)

    print(f"\n  {'':20s} {'SOLO':>10s}   {'CONCURRENT':>10s}   {'DELTA':>10s}")
    print(f"  {'─' * 55}")

    if isinstance(mlx_solo_speed, (int, float)) and isinstance(mlx_conc_speed, (int, float)) and mlx_solo_speed > 0:
        delta_mlx = ((mlx_conc_speed - mlx_solo_speed) / mlx_solo_speed) * 100
        print(f"  {'MLX (GPU)':20s} {mlx_solo_speed:>8.1f}   {mlx_conc_speed:>8.1f}   {delta_mlx:>+8.1f}%")
    else:
        print(f"  {'MLX (GPU)':20s} {str(mlx_solo_speed):>10s}   {str(mlx_conc_speed):>10s}   {'?':>10s}")

    if isinstance(ane_solo_speed, (int, float)) and isinstance(ane_conc_speed, (int, float)) and ane_solo_speed > 0:
        delta_ane = ((ane_conc_speed - ane_solo_speed) / ane_solo_speed) * 100
        print(f"  {'ANE (Neural Engine)':20s} {ane_solo_speed:>8.1f}   {ane_conc_speed:>8.1f}   {delta_ane:>+8.1f}%")
    else:
        print(f"  {'ANE (Neural Engine)':20s} {str(ane_solo_speed):>10s}   {str(ane_conc_speed):>10s}   {'?':>10s}")

    print()
    print(f"  Total concurrent wall time: {total_concurrent:.2f}s")
    print()

    if isinstance(mlx_conc_speed, (int, float)) and isinstance(ane_conc_speed, (int, float)):
        if mlx_conc_speed > 0 and ane_conc_speed > 0:
            if isinstance(delta_mlx, (int, float)) and isinstance(delta_ane, (int, float)):
                if abs(delta_mlx) < 15 and abs(delta_ane) < 15:
                    print("  ✅ HYPOTHESIS CONFIRMED: Both paths run concurrently with minimal interference!")
                elif delta_mlx < -30 or delta_ane < -30:
                    print("  ❌ HYPOTHESIS REJECTED: Significant performance degradation during concurrent execution.")
                else:
                    print("  ⚠️  PARTIAL: Some interference detected, but both paths completed.")

    # Save all results
    all_results = {
        "test_prompt": TEST_PROMPT,
        "solo": solo_results,
        "concurrent": dict(results),
        "total_concurrent_time": round(total_concurrent, 2),
        "hardware": "MacBook Air M5, 16GB",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    outfile = "/Users/midas/Desktop/cowork/dual-path-inference/dual_path_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  💾 Results saved to {outfile}")


if __name__ == "__main__":
    main()
