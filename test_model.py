"""Quick diagnostic: test if the model itself is broken or just the fine-tuned one"""
import requests
import json

SERVER = "http://localhost:5006"

def test(name, prompt, **kwargs):
    params = {
        "prompt": prompt,
        "n_predict": 50,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "stream": False,
        "cache_prompt": False,
    }
    params.update(kwargs)
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    try:
        resp = requests.post(f"{SERVER}/completion", json=params, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        content = result.get("content", "")
        print(f"OUTPUT ({len(content)} chars): {repr(content[:200])}")
        has_slash = len(content) > 10 and content.count('/') > len(content) * 0.5
        print(f"SLASH BUG: {'YES - BROKEN' if has_slash else 'NO - WORKING'}")
    except Exception as e:
        print(f"ERROR: {e}")

print("="*60)
print("TESTING CURRENT MODEL ON SERVER (port 5006)")
print("="*60)

# Simple test
test("simple_hello",
     "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n")

# Chat completions
print(f"\n{'='*60}")
print("TEST: chat_completions")
try:
    resp = requests.post(f"{SERVER}/v1/chat/completions", json={
        "model": "test",
        "messages": [{"role": "user", "content": "Hello, say hi back in one sentence"}],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False,
    }, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    msg = result['choices'][0]['message']
    content = msg.get('content', '')
    reasoning = msg.get('reasoning_content', '')
    combined = content + reasoning
    print(f"content: {repr(content[:200])}")
    print(f"reasoning: {repr(reasoning[:200])}")
    has_slash = len(combined) > 10 and combined.count('/') > len(combined) * 0.3
    has_qmark = len(combined) > 10 and combined.count('?') > len(combined) * 0.3
    print(f"BROKEN: {'YES' if has_slash or has_qmark else 'NO - WORKING'}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*60)
print("""
NEXT STEPS:
1. Stop llama-server (Ctrl+C)
2. Edit start_llama_server.bat to use the ORIGINAL model:
   Change: --model "%~dp0test\\qwen35_marine_Q4_K_M.gguf"
   To:     --model "%~dp0test\\Qwen3.5-9B-Q4_K_M.gguf"
3. Start llama-server again
4. Run this test again: python test_model.py

If original model WORKS: fine-tuned GGUF is corrupted, need to re-quantize
If original model BROKEN: llama-server build b8555 is buggy with Qwen3.5 on RTX 5060 Ti
""")
