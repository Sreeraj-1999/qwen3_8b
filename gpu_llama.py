import warnings
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.utils.transform", message=".*rcond.*")
warnings.filterwarnings("ignore", message=".*Quadro P400.*")
warnings.filterwarnings("ignore", message=".*cuda capability.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*Please install PyTorch.*")

import time
import requests as http_requests

# llama-server runs separately on port 5006 (start with start_llama_server.bat)
LLAMA_SERVER_URL = "http://localhost:5006"
LLAMA_N_CTX = 32768
logger.info("GPU LLM service will use llama-server at %s", LLAMA_SERVER_URL)

def check_llama_server_info():
    """Check llama-server version and capabilities on startup"""
    try:
        # Check /health
        resp = http_requests.get(f"{LLAMA_SERVER_URL}/health", timeout=5)
        logger.info(f"=== LLAMA-SERVER HEALTH: {resp.status_code} ===")

        # Check /props for server config
        resp2 = http_requests.get(f"{LLAMA_SERVER_URL}/props", timeout=5)
        if resp2.status_code == 200:
            props = resp2.json()
            logger.info(f"=== LLAMA-SERVER PROPS: {json.dumps(props, indent=2)[:1000]} ===")

        # Check version
        resp3 = http_requests.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=5)
        if resp3.status_code == 200:
            models = resp3.json()
            logger.info(f"=== LLAMA-SERVER MODELS: {json.dumps(models)[:500]} ===")

    except Exception as e:
        logger.warning(f"Could not reach llama-server at startup: {e} (start it with start_llama_server.bat)")

# Clear comtypes cache BEFORE importing comtypes
import shutil
import tempfile


cache_dir = os.path.join(tempfile.gettempdir(), "comtypes_cache")
if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        print(f"Cleared comtypes cache: {cache_dir}")
    except Exception as e:
        print(f"Could not clear cache: {e}")

# Now import everything else
from flask import Flask, request, jsonify, Response
import queue
import threading
import torch
from mcp_tool_handler import TELEMETRY_TOOLS, PMS_TOOLS, ALL_TOOLS, execute_tool_call, parse_tool_call, needs_tool_call

import soundfile as sf
import io
import pickle
import pandas as pd
import json
from torch import nn
import numpy as np
from comtypes.client import CreateObject
from comtypes import CoInitialize, CoUninitialize
import base64
from scipy.signal import resample
import whisper
import sys
import contextlib
import cv2
from embedding_service import embedding_service
from dotenv import load_dotenv

load_dotenv()



# Initialize Flask app for GPU operations
app = Flask('gpu_service')

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Initialize InsightFace (GPU)
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'
from insightface.app import FaceAnalysis

with suppress_stdout_stderr():
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    face_app.prepare(ctx_id=0)

# ==============================================================================
# LLM MODEL SETUP — GGUF via llama-cpp-python (replaces transformers + QLoRA)
# ==============================================================================


# We still need the tokenizer for apply_chat_template (building prompts)
from transformers import AutoTokenizer

# TOKENIZER_PATH = r"C:\Users\User\Desktop\siemens\OFFSHORE\qwen3_marine_final_v3"
TOKENIZER_PATH = r"C:\Users\User\Desktop\siemens\PROJECT5D\test\qwen35_tokenizer"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = tokenizer.eos_token

# Load Whisper model (GPU)
whisper_model = whisper.load_model("small").to("cuda")

logger.info("All GPU models loaded successfully")

# FUEL ANALYSIS - Ship Model Loading
def load_ship_model(ship_id):
    """Load model and scalers for a specific ship"""
    base_path = r'C:\Users\User\Desktop\siemens\clemens'
    model_dir = os.path.join(base_path, "model", ship_id)
    
    try:
        # Load hyperparameters
        params_path = os.path.join(model_dir, f'FULL{ship_id}_model_params.json')
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        
        # Define neural network model
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size, n_layers, n_neurons, dropout_rate, l2_regularization):
                super(NeuralNetwork, self).__init__()
                layers = []
                layers.append(nn.Linear(input_size, n_neurons))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

                for _ in range(n_layers - 1):
                    layers.append(nn.Linear(n_neurons, n_neurons))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))

                layers.append(nn.Linear(n_neurons, 1))
                self.model = nn.Sequential(*layers)
                self.l2_regularization = l2_regularization

            def forward(self, x):
                return self.model(x)
        
        # Load model with dynamic parameters
        fuel_model = NeuralNetwork(
            input_size=model_params['pca_n_components'],
            n_layers=model_params['n_layers'],
            n_neurons=model_params['n_neurons'], 
            dropout_rate=model_params['dropout_rate'],
            l2_regularization=model_params['l2_regularization']
        )
        
        model_path = os.path.join(model_dir, f'FULL{ship_id}_pytorch_model.pth')
        fuel_model.load_state_dict(torch.load(model_path))
        fuel_model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fuel_model.to(device)
        
        # Load scalers
        scaler_X_path = os.path.join(model_dir, f'FULL{ship_id}_scaler_X.pkl')
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        
        scaler_y_path = os.path.join(model_dir, f'FULL{ship_id}_scaler_y.pkl')
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        
        pca_X_path = os.path.join(model_dir, f'FULL{ship_id}_pca_X.pkl')
        with open(pca_X_path, 'rb') as f:
            pca_X = pickle.load(f)
        
        return fuel_model, scaler_X, scaler_y, pca_X, device
        
    except Exception as e:
        raise FileNotFoundError(f"Model files not found for ship {ship_id}: {str(e)}")

def relative_wind_direction(wind_direction, ship_heading):
    """Calculate relative wind direction"""
    relative_direction = wind_direction - ship_heading
    return np.mod(relative_direction + 360, 360)


# ==============================================================================
# HELPER: Generate response from GGUF model
# ==============================================================================
def gguf_generate(prompt, max_tokens=4096, temperature=0.7, stop=None,
                   top_p=0.8, top_k=20, presence_penalty=1.5, repeat_penalty=1.0):
    """Generate text via llama-server /completion endpoint. Returns the generated text."""
    if stop is None:
        stop = ["<|im_end|>", "<|endoftext|>"]

    resp = http_requests.post(f"{LLAMA_SERVER_URL}/completion", json={
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
        "stream": False,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["content"]


def gguf_stream(prompt, max_tokens=4096, temperature=0.7, stop=None,
                 top_p=0.8, top_k=20, presence_penalty=1.5, repeat_penalty=1.0):
    """Stream text via llama-server /completion endpoint. Yields chunks."""
    if stop is None:
        stop = ["<|im_end|>", "<|endoftext|>"]

    resp = http_requests.post(f"{LLAMA_SERVER_URL}/completion", json={
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
        "stream": True,
    }, stream=True, timeout=300)
    resp.raise_for_status()

    for line in resp.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data = json.loads(line_str[6:])
                if not data.get("stop", False):
                    token = data.get("content", "")
                    if token:
                        yield token

def truncate_prompt_to_fit(prompt, max_gen_tokens=1024):
    """Trim ONLY conversation history if prompt too long. NEVER touches tool results."""
    prompt_tokens = len(tokenizer.encode(prompt))
    max_prompt_tokens = LLAMA_N_CTX - max_gen_tokens

    if prompt_tokens <= max_prompt_tokens:
        return prompt

    logger.warning(f"Prompt {prompt_tokens} tokens, limit {max_prompt_tokens}. Trimming history.")

    history_start = prompt.find("=== CONVERSATION HISTORY")
    history_end = prompt.find("=== END HISTORY ===")

    if history_start != -1 and history_end != -1:
        trimmed = prompt[:history_start] + prompt[history_end + len("=== END HISTORY ==="):]
        logger.info(f"Removed history. Now {len(tokenizer.encode(trimmed))} tokens.")
        return trimmed

    return prompt


@app.route('/gpu/llm/generate', methods=['POST'])
def generate_llm_response():
    """Generate LLM response — with automatic tool calling"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        response_type = data.get('response_type', 'general')
        enable_thinking = data.get('enable_thinking', False)
        imo = data.get('imo')
        if imo and messages and messages[0].get('role') == 'system':
            messages[0]['content'] += f""" You are currently monitoring vessel IMO {imo}. Always use this IMO when calling tools.
            Current year is 2026. When querying time ranges, use 2026.
            When reporting vessel position always use the 'current_location' field (human readable place name) instead of raw latitude/longitude coordinates.
            Key sensor names: ME_RPM, SA_POW_act_kW@AVG (shaft power), ME_FMS_act_kgPh@AVG (ME fuel consumption), AE_FMS_act_kgPh@AVG (AE fuel), V_SOG_act_kn@AVG (speed), ME_Load@AVG (ME load %), ME_SCAV_AIR_PRESS (scav air pressure), ME_NO_1_TC_RPM (TC1 RPM), ME_NO_2_TC_RPM (TC2 RPM). VesselTimeStamp is unix epoch."""
            # messages[0]['content'] += f""" You are currently monitoring vessel IMO {imo}. Always use this IMO when calling tools.
            # Current year is 2026. When querying time ranges, use 2026.
            # When reporting vessel position always use the 'current_location' field (human readable place name) instead of raw latitude/longitude coordinates.
            # Key sensor names: ME_RPM, SA_POW_act_kW@AVG (shaft power), ME_FMS_act_kgPh@AVG (ME fuel consumption), AE_FMS_act_kgPh@AVG (AE fuel), V_SOG_act_kn@AVG (speed), ME_Load@AVG (ME load %), ME_SCAV_AIR_PRESS (scav air pressure), ME_NO_1_TC_RPM (TC1 RPM), ME_NO_2_TC_RPM (TC2 RPM). VesselTimeStamp is unix epoch."""
            # messages[0]['content'] += f""" You are currently monitoring vessel IMO {imo}. Always use this IMO when calling tools.
            # Current year is 2026. When querying time ranges, use 2026.
            # Key sensor names: ME_RPM, SA_POW_act_kW@AVG (shaft power), ME_FMS_act_kgPh@AVG (ME fuel consumption), AE_FMS_act_kgPh@AVG (AE fuel), V_SOG_act_kn@AVG (speed), ME_Load@AVG (ME load %), ME_SCAV_AIR_PRESS (scav air pressure), ME_NO_1_TC_RPM (TC1 RPM), ME_NO_2_TC_RPM (TC2 RPM). VesselTimeStamp is unix epoch."""

        if not messages:
            return jsonify({'error': 'No messages provided'}), 400

        # Tools always available based on context
        tools = ALL_TOOLS if imo else PMS_TOOLS

        # Separate system msg, history msgs, and current user msg
        system_msg = None
        history_msgs = []
        current_user_msg = None
        
        for m in messages:
            if m['role'] == 'system' and system_msg is None:
                system_msg = m
            elif m['role'] == 'user' and m == messages[-1]:
                current_user_msg = m
            else:
                history_msgs.append(m)
        
        # If there's history, bake it into the system prompt
        if history_msgs and system_msg:
            history_text = "\n\n=== CONVERSATION HISTORY (use this for follow-up questions) ===\n"
            for h in history_msgs:
                role_label = "User" if h['role'] == 'user' else "Assistant"
                content = h['content'][:300] + "..." if len(h['content']) > 300 else h['content']
                history_text += f"{role_label}: {content}\n"
            history_text += "=== END HISTORY ===\n"
            system_msg['content'] += history_text
            
            messages = [system_msg, current_user_msg]
        
        logger.info(f"=== PROMPT: {len(messages)} msgs, tools={'yes' if tools else 'no'}, history={len(history_msgs)} msgs ===")

        # Build prompt using tokenizer's chat template — always disable thinking
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=tools
        )
        # Strip <think> tags — model generates "/" when it sees them
        text = text.replace('<think>\n\n</think>\n\n', '')
        text = text.replace('<think>', '').replace('</think>', '')

        max_tok = data.get('max_tokens', 4096)
        gen_temp = 0.7
        response = gguf_generate(text, max_tokens=max_tok, temperature=gen_temp)
        if "<tool_call>" in response:
            tool_name, tool_args = parse_tool_call(response)
            if tool_name:
                tool_result = execute_tool_call(tool_name, tool_args)
                
                with open('tool_debug.txt', 'w', encoding='utf-8') as f:
                    f.write("=== MODEL FIRST RESPONSE ===\n")
                    f.write(response + "\n\n")
                    f.write("=== TOOL CALLED ===\n")
                    f.write(f"Name: {tool_name}\n")
                    f.write(f"Args: {json.dumps(tool_args, indent=2)}\n\n")
                    f.write("=== TOOL RESULT ===\n")
                    f.write(tool_result + "\n\n")
                logger.info(f"Tool debug written to tool_debug.txt")

                messages.append({"role": "assistant", "content": response.replace("<|im_end|>", "").strip()})
                messages.append({"role": "user", "content": f"Tool result:\n{tool_result}\n\nAnswer the original question based on this data. List ALL items completely."})

                text2 = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False
                )
                text2 = text2.replace('<think>\n\n</think>\n\n', '')
                text2 = text2.replace('<think>', '').replace('</think>', '')
                text2 = truncate_prompt_to_fit(text2, max_gen_tokens=4096)
                response = gguf_generate(text2, max_tokens=4096, temperature=0.7)

        # Tool call loop (max 1 round)
        # if "<tool_call>" in response:
        #     tool_name, tool_args = parse_tool_call(response)
        #     if tool_name:
        #         tool_result = execute_tool_call(tool_name, tool_args)
        #         with open('tool_debug.txt', 'w', encoding='utf-8') as f:
        #             f.write("=== MODEL FIRST RESPONSE ===\n")
        #             f.write(response + "\n\n")
        #             f.write("=== TOOL CALLED ===\n")
        #             f.write(f"Name: {tool_name}\n")
        #             f.write(f"Args: {json.dumps(tool_args, indent=2)}\n\n")
        #             f.write("=== TOOL RESULT ===\n")
        #             f.write(tool_result + "\n\n")
        #         logger.info(f"Tool debug written to tool_debug.txt")

        #         messages.append({"role": "assistant", "content": response.replace("<|im_end|>", "").strip()})
        #         messages.append({"role": "user", "content": f"Tool result:\n{tool_result}\n\nAnswer the original question based on this data."})

        #         text2 = tokenizer.apply_chat_template(
        #             messages, tokenize=False,
        #             add_generation_prompt=True, enable_thinking=False
        #         )

        #         response = gguf_generate(text2, max_tokens=1024, temperature=0.7)

        # Clean
        if "</think>" in response:
            response = response.split("</think>")[-1]
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

        return jsonify({'response': response, 'response_type': response_type, 'status': 'success'})

    except Exception as e:
        logger.error(f"Error in LLM generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/gpu/llm/stream', methods=['POST'])
def stream_llm_response():
    """Stream LLM response via /v1/chat/completions — server handles reasoning budget"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])

        if not messages:
            return jsonify({'error': 'No messages provided'}), 400

        def generate():
            try:
                # --- History handling ---
                system_msg = None
                history_msgs = []
                current_user_msg = None

                for m in messages:
                    if m['role'] == 'system' and system_msg is None:
                        system_msg = m
                    elif m['role'] == 'user' and m == messages[-1]:
                        current_user_msg = m
                    else:
                        history_msgs.append(m)

                if history_msgs and system_msg:
                    history_text = "\n\n=== CONVERSATION HISTORY (use this for follow-up questions) ===\n"
                    for h in history_msgs:
                        role_label = "User" if h['role'] == 'user' else "Assistant"
                        content = h['content'][:300] + "..." if len(h['content']) > 300 else h['content']
                        history_text += f"{role_label}: {content}\n"
                    history_text += "=== END HISTORY ===\n"
                    system_msg['content'] += history_text

                    final_messages = [system_msg, current_user_msg]
                else:
                    final_messages = messages

                logger.info(f"=== STREAM: {len(final_messages)} msgs, history={len(history_msgs)} msgs ===")

                # ============================================================
                # Use /completion with tokenizer — enable_thinking=True
                # Real-time streaming: thinking tokens → thinking box,
                # after </think> → answer box
                # Qwen3.5 official params: temp=1.0, top_p=0.95 for thinking
                # ============================================================
                stream_start_time = time.time()

                text = tokenizer.apply_chat_template(
                    final_messages, tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )

                logger.info(f"=== STREAM PROMPT (last 200): {repr(text[-200:])} ===")

                # Real-time stream with state tracking
                in_thinking = True  # Model starts in thinking mode after <think>
                thinking_buf = ""
                answer_buf = ""
                raw_buf = ""         # For detecting </think> boundary

                for token in gguf_stream(text, max_tokens=4096, temperature=1.0,
                                          top_p=0.95, top_k=20, presence_penalty=1.5):
                    raw_buf += token

                    if in_thinking:
                        # Check if </think> appeared
                        if '</think>' in raw_buf:
                            # Split at </think>
                            parts = raw_buf.split('</think>', 1)
                            # Send remaining thinking content
                            think_text = parts[0].replace('<think>', '')
                            if think_text:
                                thinking_buf += think_text
                                yield f"data: {json.dumps({'type': 'thinking', 'content': think_text})}\n\n"
                            # Switch to answer mode
                            in_thinking = False
                            raw_buf = parts[1]  # Leftover after </think>
                            if raw_buf.strip():
                                clean = raw_buf.replace('<|im_end|>', '').replace('<|im_start|>', '')
                                if clean.strip():
                                    answer_buf += clean
                                    yield f"data: {json.dumps({'type': 'answer', 'content': clean})}\n\n"
                            raw_buf = ""
                        else:
                            # Still thinking — stream tokens to thinking box
                            # But buffer a bit to avoid sending partial </think>
                            safe_len = len(raw_buf) - 8  # Keep last 8 chars as buffer
                            if safe_len > 0:
                                to_send = raw_buf[:safe_len].replace('<think>', '')
                                raw_buf = raw_buf[safe_len:]
                                if to_send:
                                    thinking_buf += to_send
                                    # Cap thinking display at 3000 chars
                                    if len(thinking_buf) <= 3000:
                                        yield f"data: {json.dumps({'type': 'thinking', 'content': to_send})}\n\n"
                    else:
                        # In answer mode — stream directly
                        clean = token.replace('<|im_end|>', '').replace('<|im_start|>', '')
                        clean = clean.replace('<think>', '').replace('</think>', '')
                        if clean:
                            answer_buf += clean
                            yield f"data: {json.dumps({'type': 'answer', 'content': clean})}\n\n"

                # Flush remaining buffer
                if in_thinking and raw_buf:
                    # Never got </think> — model used all tokens on thinking
                    think_text = raw_buf.replace('<think>', '').replace('</think>', '')
                    thinking_buf += think_text
                    if think_text.strip() and len(thinking_buf) <= 3000:
                        yield f"data: {json.dumps({'type': 'thinking', 'content': think_text})}\n\n"

                elapsed = time.time() - stream_start_time
                logger.info(f"=== STREAM DONE: thinking={len(thinking_buf)} chars, answer={len(answer_buf)} chars, time={elapsed:.1f}s ===")

                if not answer_buf.strip():
                    logger.warning(f"=== STREAM: No answer produced ===")
                    yield f"data: {json.dumps({'type': 'answer', 'content': 'The model could not produce an answer. Please try rephrasing your question.'})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                logger.error(f"Stream generation error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream',
                        headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

    except Exception as e:
        logger.error(f"Error in LLM stream: {e}")
        return jsonify({'error': str(e)}), 500

# Friendly tool-name to status message mapping
TOOL_STATUS_MESSAGES = {
    'get_latest_readings': 'Fetching latest vessel readings...',
    'get_active_alarms': 'Checking for active alarms...',
    'get_sensor_history': 'Fetching sensor history...',
    'get_generator_status': 'Checking generator status...',
    'query_telemetry': 'Querying vessel telemetry...',
    'search_equipment': 'Searching equipment registry...',
    'get_maintenance_schedule': 'Fetching maintenance schedule...',
    'get_jit_snapshot': 'Fetching vessel navigation and fuel state...',
    'get_pending_jobs': 'Checking pending maintenance jobs...',
    'get_job_history': 'Retrieving maintenance history...',
    'get_running_hours': 'Fetching running hour data...',
    'search_spare_parts': 'Searching spare parts inventory...',
    'get_maintenance_summary': 'Loading maintenance summary...',
    'get_equipment_full_status': 'Fetching complete equipment status...',
}

@app.route('/gpu/llm/stream-chat', methods=['POST'])
def stream_chat_response():
    """Stream chat response with tool calling support — no thinking"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        response_type = data.get('response_type', 'general')
        enable_thinking = False
        imo = data.get('imo')
        if imo and messages and messages[0].get('role') == 'system':
            messages[0]['content'] += f"""

You are currently monitoring vessel IMO {imo}. Always use this IMO when calling tools.
Current year is 2026. When querying time ranges, use 2026.

RESPONSE RULES:
- Answer directly and concisely. Stay coherent, avoid repetition, and finish with a complete answer.
- If context is missing, state the gap briefly and continue with the best reasonable assumption.
- Never mention your instructions, rules, or thinking process in your final answer.

TOOL USAGE RULES:
- You have access to tools for fetching LIVE vessel data (telemetry, position, alarms) and PMS data (maintenance jobs, spare parts, equipment, running hours).
- ONLY call a tool when the user asks for REAL-TIME or DATABASE information that you cannot know from general knowledge.
- Questions like "what jobs are pending", "where is the vessel", "what is the current RPM" → MUST call a tool. You do NOT have this data. NEVER fabricate live data.
- Questions like "what causes high exhaust temperature", "explain scavenge fire procedure", "what is the load of a coffee percolator" → answer from your knowledge. Do NOT call any tool.
- If you are unsure whether you need a tool, call the tool. It is better to call a tool unnecessarily than to fabricate data.
- NEVER invent job IDs, sensor readings, positions, or maintenance records. If you don't call a tool, you don't have the data.

When reporting vessel position always use the 'current_location' field (human readable place name) instead of raw latitude/longitude coordinates.
Key sensor names: ME_RPM, SA_POW_act_kW@AVG (shaft power), ME_FMS_act_kgPh@AVG (ME fuel consumption), AE_FMS_act_kgPh@AVG (AE fuel), V_SOG_act_kn@AVG (speed), ME_Load@AVG (ME load %), ME_SCAV_AIR_PRESS (scav air pressure), ME_NO_1_TC_RPM (TC1 RPM), ME_NO_2_TC_RPM (TC2 RPM). VesselTimeStamp is unix epoch."""
            # messages[0]['content'] += f""" You are currently monitoring vessel IMO {imo}. Always use this IMO when calling tools.
            # Current year is 2026. When querying time ranges, use 2026.
            # Key sensor names: ME_RPM, SA_POW_act_kW@AVG (shaft power), ME_FMS_act_kgPh@AVG (ME fuel consumption), AE_FMS_act_kgPh@AVG (AE fuel), V_SOG_act_kn@AVG (speed), ME_Load@AVG (ME load %), ME_SCAV_AIR_PRESS (scav air pressure), ME_NO_1_TC_RPM (TC1 RPM), ME_NO_2_TC_RPM (TC2 RPM). VesselTimeStamp is unix epoch."""

        if not messages:
            return jsonify({'error': 'No messages provided'}), 400

        # Tools always available based on context
        tools = ALL_TOOLS if imo else PMS_TOOLS

        # --- Same history handling as /generate ---
        system_msg = None
        history_msgs = []
        current_user_msg = None

        for m in messages:
            if m['role'] == 'system' and system_msg is None:
                system_msg = m
            elif m['role'] == 'user' and m == messages[-1]:
                current_user_msg = m
            else:
                history_msgs.append(m)

        if history_msgs and system_msg:
            history_text = "\n\n=== CONVERSATION HISTORY (use this for follow-up questions) ===\n"
            for h in history_msgs:
                role_label = "User" if h['role'] == 'user' else "Assistant"
                content = h['content'][:300] + "..." if len(h['content']) > 300 else h['content']
                history_text += f"{role_label}: {content}\n"
            history_text += "=== END HISTORY ===\n"
            system_msg['content'] += history_text
            messages = [system_msg, current_user_msg]

        logger.info(f"=== STREAM-CHAT PROMPT: {len(messages)} msgs, tools={'yes' if tools else 'no'}, history={len(history_msgs)} msgs ===")

        def generate():
            try:
                user_question = ""
                for m in reversed(messages):
                    if m['role'] == 'user':
                        user_question = m['content']
                        break

                likely_tool = bool(tools)
                logger.info(f"DEBUG TOOL - user_question: '{user_question[:100]}', tools: {bool(tools)}, likely_tool: {likely_tool}")

                if likely_tool:
                    logger.info(f"DEBUG PATH A - generating with tool detection")
                    # === PATH A: Tool detection — /completion with enable_thinking=False ===
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        tools=tools
                    )
                    # Strip <think> tags — model generates "/" when it sees them
                    text = text.replace('<think>\n\n</think>\n\n', '')
                    text = text.replace('<think>', '').replace('</think>', '')
                    response = gguf_generate(text, max_tokens=4096, temperature=0.7)

                    if "<tool_call>" in response:
                        tool_name, tool_args = parse_tool_call(response)
                        if tool_name:
                            # Send friendly status message
                            status_msg = TOOL_STATUS_MESSAGES.get(tool_name, 'Processing your request...')
                            yield f"data: {json.dumps({'type': 'tool_status', 'content': status_msg})}\n\n"

                            # Execute the tool
                            tool_result = execute_tool_call(tool_name, tool_args)

                            with open('tool_debug.txt', 'w', encoding='utf-8') as f:
                                f.write("=== MODEL FIRST RESPONSE ===\n")
                                f.write(response + "\n\n")
                                f.write("=== TOOL CALLED ===\n")
                                f.write(f"Name: {tool_name}\n")
                                f.write(f"Args: {json.dumps(tool_args, indent=2)}\n\n")
                                f.write("=== TOOL RESULT ===\n")
                                f.write(tool_result + "\n\n")
                            logger.info(f"Tool debug written to tool_debug.txt")

                            # Build Pass 2 messages
                            pass2_messages = list(messages)
                            pass2_messages.append({"role": "assistant", "content": response.replace("<|im_end|>", "").strip()})
                            pass2_messages.append({"role": "user", "content": f"Tool result:\n{tool_result}\n\nAnswer the original question based on this data. List ALL items completely."})

                            # === PASS 2: Stream the final answer (no thinking) ===
                            text2 = tokenizer.apply_chat_template(
                                pass2_messages, tokenize=False,
                                add_generation_prompt=True, enable_thinking=False
                            )
                            # Strip <think> tags
                            text2 = text2.replace('<think>\n\n</think>\n\n', '')
                            text2 = text2.replace('<think>', '').replace('</think>', '')
                            text2 = truncate_prompt_to_fit(text2, max_gen_tokens=4096)

                            # Collect then send — strips all think tags reliably
                            pass2_buffer = ""
                            for token in gguf_stream(text2, max_tokens=4096, temperature=0.7):
                                pass2_buffer += token

                            # Strip any thinking that leaked despite enable_thinking=False
                            if "</think>" in pass2_buffer:
                                pass2_buffer = pass2_buffer.split("</think>")[-1]
                            pass2_buffer = pass2_buffer.replace("<think>", "").replace("</think>", "")
                            pass2_buffer = pass2_buffer.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

                            # Stream cleaned answer in chunks
                            chunk_size = 6
                            for i in range(0, len(pass2_buffer), chunk_size):
                                chunk = pass2_buffer[i:i + chunk_size]
                                if chunk:
                                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"
                                    time.sleep(0.02)

                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                            return

                    # Tool was expected but model didn't call one — send response as stream
                    if "</think>" in response:
                        response = response.split("</think>")[-1]
                    response = response.replace("<think>", "").replace("</think>", "")
                    response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

                    # Send in reasonable chunks
                    chunk_size = 6
                    for i in range(0, len(response), chunk_size):
                        chunk = response[i:i + chunk_size]
                        if chunk:
                            yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"
                            time.sleep(0.03)

                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

                else:
                    logger.info(f"DEBUG PATH B - direct streaming, no tools")
                    # === PATH B: No tools — collect + strip thinking ===
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    # Strip <think> tags
                    text = text.replace('<think>\n\n</think>\n\n', '')
                    text = text.replace('<think>', '').replace('</think>', '')

                    pathb_buffer = ""
                    for token in gguf_stream(text, max_tokens=4096, temperature=0.7):
                        pathb_buffer += token

                    if "</think>" in pathb_buffer:
                        pathb_buffer = pathb_buffer.split("</think>")[-1]
                    pathb_buffer = pathb_buffer.replace("<think>", "").replace("</think>", "")
                    pathb_buffer = pathb_buffer.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

                    chunk_size = 6
                    for i in range(0, len(pathb_buffer), chunk_size):
                        chunk = pathb_buffer[i:i + chunk_size]
                        if chunk:
                            yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"
                            time.sleep(0.02)

                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                logger.error(f"Stream-chat error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream',
                        headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

    except Exception as e:
        logger.error(f"Error in stream-chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/gpu/llm/vision', methods=['POST'])
def vision_query():
    """Analyze an image using Qwen3.5 vision via llama-server"""
    try:
        data = request.get_json()
        image_b64 = data.get('image')  # base64 encoded image
        question = data.get('question', 'What do you see in this image?')
        max_tokens = data.get('max_tokens', 4096)

        if not image_b64:
            return jsonify({'error': 'No image provided (base64)'}), 400

        logger.info(f"Vision query: {question[:80]}")

        # Call llama-server's OpenAI-compatible vision endpoint
        resp = http_requests.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json={
            'model': 'qwen3.5-9b',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a marine engineering diagram analyst. Study the image carefully. Give a direct, specific answer in 1-4 sentences. No preamble, no over-explanation.'
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                        {'type': 'text', 'text': question}
                    ]
                }
            ],
            'max_tokens': max_tokens,
            'temperature': 0.6,
            'top_p': 0.8,
            'top_k': 20,
            'presence_penalty': 1.5,
            'repeat_penalty': 1.0,
        }, timeout=300)
        resp.raise_for_status()

        result = resp.json()
        msg = result['choices'][0]['message']
        content_text = msg.get('content', '')
        reasoning_text = msg.get('reasoning_content', '')

        # Combine — model may put everything in either field
        combined = content_text if content_text and content_text.strip() else reasoning_text

        # Extract answer: everything after </think>, or the whole thing if no think tags
        if '</think>' in combined:
            answer = combined.split('</think>')[-1].strip()
        elif '<think>' in combined:
            # Started thinking but never finished — try last few lines
            lines = combined.replace('<think>', '').strip().split('\n')
            answer = '\n'.join(lines[-3:]).strip() if len(lines) > 3 else combined.replace('<think>', '').strip()
        else:
            answer = combined

        # Clean up
        answer = answer.replace('<|im_end|>', '').replace('<|im_start|>', '')
        answer = answer.replace('<think>', '').replace('</think>', '').strip()

        return jsonify({'response': answer, 'status': 'success'})

    except Exception as e:
        logger.error(f"Vision query error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/gpu/llm/vision/stream', methods=['POST'])
def vision_query_stream():
    """Stream vision analysis with thinking/answer separation"""
    try:
        data = request.get_json()
        image_b64 = data.get('image')
        question = data.get('question', 'What do you see in this image?')
        max_tokens = data.get('max_tokens', 4096)

        if not image_b64:
            return jsonify({'error': 'No image provided (base64)'}), 400

        logger.info(f"Vision stream query: {question[:80]}")

        def generate():
            try:
                resp = http_requests.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json={
                    'model': 'qwen3.5-9b',
                    'messages': [
                        {
                            'role': 'system',
                            'content': 'You are a marine engineering diagram analyst. Study the image carefully. Give a direct, specific answer in 1-4 sentences. No preamble, no over-explanation.'
                        },
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                                {'type': 'text', 'text': question}
                            ]
                        }
                    ],
                    'max_tokens': max_tokens,
                    'temperature': 0.6,
                    'top_p': 0.8,
                    'top_k': 20,
                    'presence_penalty': 1.5,
                    'repeat_penalty': 1.0,
                    'stream': True,
                }, stream=True, timeout=300)
                resp.raise_for_status()

                # Collect the FULL response first, then parse
                # Qwen3.5 vision: content may be empty, everything in reasoning_content
                # OR content has <think>...</think> then answer
                # OR reasoning_content has thinking, content has answer
                full_text = ""           # Everything from content field
                reasoning_text = ""      # Everything from reasoning_content field

                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode('utf-8')
                    if not line_str.startswith('data: '):
                        continue
                    if line_str.strip() == 'data: [DONE]':
                        break

                    chunk = json.loads(line_str[6:])
                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')
                    reasoning = delta.get('reasoning_content', '')

                    if content:
                        full_text += content
                    if reasoning:
                        reasoning_text += reasoning

                logger.info(f"VISION RAW: content={len(full_text)} chars, reasoning={len(reasoning_text)} chars")

                # === EXTRACT ANSWER from wherever the model put it ===
                # Combine everything into one blob and parse
                combined = ""
                if full_text.strip():
                    combined = full_text
                elif reasoning_text.strip():
                    combined = reasoning_text

                # Now split thinking from answer
                thinking_part = ""
                answer_part = ""

                if '</think>' in combined:
                    # Clean split — everything before </think> is thinking, after is answer
                    parts = combined.split('</think>', 1)
                    thinking_part = parts[0].replace('<think>', '').strip()
                    answer_part = parts[1].strip()
                elif '<think>' in combined:
                    # Has <think> but no </think> — model ran out of tokens mid-thinking
                    # Try to find the last coherent sentence as answer
                    without_think = combined.replace('<think>', '')
                    # Look for the last paragraph break or sentence
                    lines = without_think.strip().split('\n')
                    if len(lines) > 3:
                        # Last few lines might be the answer attempt
                        answer_part = '\n'.join(lines[-3:]).strip()
                        thinking_part = '\n'.join(lines[:-3]).strip()
                    else:
                        answer_part = without_think.strip()
                else:
                    # No think tags at all — entire thing is the answer
                    answer_part = combined.strip()

                # Clean up special tokens
                answer_part = answer_part.replace('<|im_end|>', '').replace('<|im_start|>', '')
                answer_part = answer_part.replace('<think>', '').replace('</think>', '').strip()

                logger.info(f"VISION FINAL: thinking={len(thinking_part)} chars, answer={len(answer_part)} chars")
                logger.info(f"VISION ANSWER: {answer_part[:300]}")

                # Send thinking to thinking box
                if thinking_part:
                    yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_part[:2000]})}\n\n"

                # Stream the answer in chunks
                if answer_part:
                    chunk_size = 20
                    for i in range(0, len(answer_part), chunk_size):
                        yield f"data: {json.dumps({'type': 'answer', 'content': answer_part[i:i+chunk_size]})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'answer', 'content': 'Could not analyze the image. Please try again with a clearer image.'})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                logger.error(f"Vision stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream',
                        headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

    except Exception as e:
        logger.error(f"Error in vision stream: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/gpu/stt/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Whisper model"""
    try:
        audio_file = request.files.get('audio') or request.files.get('file')
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400
        
        logger.info("Starting audio transcription with Whisper")
        
        # Read audio data
        audio_data = audio_file.read()

        from pydub import AudioSegment
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            audio_data = wav_buffer.read()
        except Exception as conv_err:
            logger.warning(f"Audio format conversion: {conv_err}, trying raw")
        
        # Convert to numpy array
        audio_np, sample_rate = sf.read(io.BytesIO(audio_data))
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            target_length = int(len(audio_np) * 16000 / sample_rate)
            audio_np = resample(audio_np, target_length)
            sample_rate = 16000
        
        # Ensure mono audio
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        
        audio_np = audio_np.astype(np.float32)
        result = whisper_model.transcribe(audio_np, language='en')
        transcription = result['text'].strip()
        
        logger.info(f"Transcription successful: {transcription[:50]}...")
        
        return jsonify({
            'transcription': transcription,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in audio transcription: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        torch.cuda.empty_cache()

@app.route('/gpu/tts/generate', methods=['POST'])
def text_to_speech():
    """Convert text to speech using Windows SAPI"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info("Converting text to speech")
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav.name
        temp_wav.close()

        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_mp3_path = temp_mp3.name
        temp_mp3.close()

        try:
            # Initialize COM for this thread
            CoInitialize()
            
            import comtypes.gen
            import comtypes
            
            try:
                if hasattr(comtypes.gen, '__file__') and comtypes.gen.__file__ is not None:
                    gen_dir = os.path.dirname(comtypes.gen.__file__)
                else:
                    comtypes_path = os.path.dirname(comtypes.__file__)
                    gen_dir = os.path.join(comtypes_path, 'gen')
                
                if os.path.exists(gen_dir):
                    cache_files = [f for f in os.listdir(gen_dir) 
                                 if f.startswith('_') and f.endswith('.py')]
                    for file in cache_files:
                        try:
                            file_path = os.path.join(gen_dir, file)
                            os.remove(file_path)
                            logger.info(f"Cleared cache file: {file}")
                        except Exception as e:
                            logger.warning(f"Could not remove {file}: {e}")
                else:
                    logger.warning(f"Gen directory not found at {gen_dir}")
            except Exception as e:
                logger.warning(f"Could not clear cache: {e}")
            
            from comtypes.client import GetModule
            GetModule("C:\\Windows\\System32\\Speech\\Common\\sapi.dll")
            
            from comtypes.gen import SpeechLib
            
            engine = CreateObject("SAPI.SpVoice")
            stream = CreateObject("SAPI.SpFileStream")

            voices = engine.GetVoices()
            for i in range(voices.Count):
                voice = voices.Item(i)
                voice_name = voice.GetAttribute('Name').lower()
                if 'english' in voice_name or 'en_' in voice_name or 'david' in voice_name or 'zira' in voice_name:
                    engine.Voice = voice
                    logger.info(f"Selected voice: {voice_name}")
                    break

            stream.Open(temp_wav_path, SpeechLib.SSFMCreateForWrite)
            engine.AudioOutputStream = stream
            engine.Rate = 1
            engine.Speak(text)
            stream.Close()

            from pydub import AudioSegment
            audio = AudioSegment.from_wav(temp_wav_path)
            audio.export(temp_mp3_path, format="mp3")

            with open(temp_mp3_path, 'rb') as f:
                audio_data = f.read()
                audio_blob = base64.b64encode(audio_data).decode('utf-8')

            logger.info("TTS conversion successful")
            return jsonify({
                'audio_blob': audio_blob,
                'status': 'success'
            })

        except Exception as e:
            logger.error(f"TTS conversion failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
        finally:
            try:
                CoUninitialize()
            except:
                pass
            for path in [temp_wav_path, temp_mp3_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError as e:
                    logger.warning(f"Could not remove temp file {path}: {e}")

    except Exception as e:
        logger.error(f"Error in TTS endpoint: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        torch.cuda.empty_cache() 

@app.route('/gpu/face/compare', methods=['POST'])
def compare_faces():
    """Compare two faces using InsightFace"""
    try:
        if 'image' not in request.files or 'profilePicture' not in request.files:
            return jsonify({'error': 'Both images required'}), 400

        image_file1 = request.files['image']
        image_file2 = request.files['profilePicture']

        image1 = cv2.imdecode(np.frombuffer(image_file1.read(), np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(image_file2.read(), np.uint8), cv2.IMREAD_COLOR)

        if image1 is None or image2 is None:
            return jsonify({'error': 'Invalid image format'}), 400

        faces1 = face_app.get(image1)
        faces2 = face_app.get(image2)

        if not faces1 or not faces2:
            return jsonify({'error': 'No face detected'}), 400

        encoding1 = faces1[0].embedding
        encoding2 = faces2[0].embedding

        encoding1 = encoding1 / np.linalg.norm(encoding1)
        encoding2 = encoding2 / np.linalg.norm(encoding2)

        cosine_similarity = np.dot(encoding1, encoding2)
        similarity = (cosine_similarity + 1) / 2

        logger.info(f"Face similarity: {similarity:.4f}")
        
        success = float(similarity) * 100 > 70
        
        return jsonify({
            "success": success,
            "score": str(float(similarity) * 100) if success else "0",
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in face comparison: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        torch.cuda.empty_cache() 

@app.route('/gpu/fuel/analyze', methods=['POST'])
def fuel_analysis():
    """Perform fuel consumption analysis for specific vessel"""
    try:
        data = request.get_json()
        fuel_payload = data.get('fuelmasterPayload')
        
        if not fuel_payload:
            return jsonify({'error': 'No fuel payload provided'}), 400

        imo_number = fuel_payload.get('IMO')
        if not imo_number:
            return jsonify({'error': 'IMO is required'}), 400
        
        ship_id = f"IMO{imo_number}" 
        
        try:
            fuel_model, scaler_X, scaler_y, pca_X, device = load_ship_model(ship_id)
            logger.info("Fuel model loaded successfully")
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        
        sog = fuel_payload.get('V_SOG_act_kn@AVG')
        stw = fuel_payload.get('V_STW_act_kn@AVG')
        rpm = fuel_payload.get('SA_SPD_act_rpm@AVG')
        torque = fuel_payload.get('SA_TQU_act_kNm@AVG')
        power = fuel_payload.get('SA_POW_act_kW@AVG')
        actual_fuel = fuel_payload.get('ME_FMS_act_kgPh@AVG')
        wind_direction = fuel_payload.get('WEA_WDT_act_deg@AVG')
        ship_heading = fuel_payload.get('V_HDG_act_deg@AVG')
        wind_speed = fuel_payload.get('WEA_WST_act_kn@AVG')
        
        required_fields = [sog, stw, rpm, torque, power, wind_direction, ship_heading, wind_speed]
        if any(x is None for x in required_fields):
            return jsonify({'error': 'Missing required input fields'}), 400
        
        rel_wind_dir = relative_wind_direction(wind_direction, ship_heading)
        rel_wind_cos = np.cos(np.radians(rel_wind_dir))
        effective_wind_speed = wind_speed * rel_wind_cos
        
        input_data = pd.DataFrame({
            'V_SOG_act_kn@AVG': [sog],
            'V_STW_act_kn@AVG': [stw], 
            'SA_SPD_act_rpm@AVG': [rpm],
            'SA_TQU_act_kNm@AVG': [torque],
            'Effective_Wind_Speed': [effective_wind_speed],
            'SA_POW_act_kW@AVG': [power]
        })
        
        X_scaled = scaler_X.transform(input_data)
        X_pca = pca_X.transform(X_scaled)
        X_tensor = torch.tensor(X_pca, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            y_pred_scaled = fuel_model(X_tensor).cpu().numpy().flatten()
        predicted_fuel = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
        
        alert = False
        if actual_fuel is not None and actual_fuel > 0:
            percentage_diff = abs(predicted_fuel - actual_fuel) / actual_fuel * 100
            alert = percentage_diff > 4.0
        else:
            fuel_threshold = 175
            alert = predicted_fuel > fuel_threshold
        
        flag = 1 if alert else 0
        
        return jsonify({
            'predicted': round(float(predicted_fuel), 2),
            'alert': flag,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in fuel analysis: {e}")
        return jsonify({
            "error": str(e),
            'predicted': 0,
            'alert': 1,
            'status': 'error'
        }), 200
    finally:
        torch.cuda.empty_cache() 

@app.route('/gpu/health', methods=['GET'])
def health_check():
    """Health check for GPU service"""
    try:
        gpu_available = torch.cuda.is_available()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if gpu_available else 0
        
        return jsonify({
            'status': 'healthy',
            'gpu_available': gpu_available,
            'gpu_memory_gb': round(gpu_memory / (1024**3), 2) if gpu_memory else 0,
            'models_loaded': {
                'llm': True,  # llama-server runs separately
                'whisper': whisper_model is not None,
                'face_recognition': face_app is not None
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/gpu/debug', methods=['GET'])
def debug_info():
    """Debug endpoint — shows llama-server version, config, and reasoning-budget status"""
    info = {'gpu_llama_status': 'running'}
    try:
        resp = http_requests.get(f"{LLAMA_SERVER_URL}/health", timeout=5)
        info['llama_server_health'] = resp.status_code
    except:
        info['llama_server_health'] = 'unreachable'

    try:
        resp2 = http_requests.get(f"{LLAMA_SERVER_URL}/props", timeout=5)
        info['llama_server_props'] = resp2.json() if resp2.status_code == 200 else 'N/A'
    except:
        info['llama_server_props'] = 'error'

    try:
        resp3 = http_requests.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=5)
        info['llama_server_models'] = resp3.json() if resp3.status_code == 200 else 'N/A'
    except:
        info['llama_server_models'] = 'error'

    return jsonify(info)

# ============= STARTUP =============
if __name__ == '__main__':
    logger.info("Starting GPU Service (GGUF mode)...")
    check_llama_server_info()
    app.run(host='0.0.0.0', port=5005, debug=False)