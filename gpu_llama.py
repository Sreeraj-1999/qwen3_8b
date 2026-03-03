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

logger.info("Loading GGUF model for LLM operations...")

from llama_cpp import Llama

GGUF_MODEL_PATH = r"C:\Users\User\Desktop\siemens\PROJECT5D\test\qwen3_marine_Q4_K_M.gguf"

llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_ctx=16384,
    n_gpu_layers=99,
    main_gpu=0,
    cache_type_k="q4_0",
    cache_type_v="q4_0",
    verbose=False,
)
logger.info("GGUF model loaded successfully")

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

TOKENIZER_PATH = r"C:\Users\User\Desktop\siemens\OFFSHORE\qwen3_marine_final_v3"
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
def gguf_generate(prompt, max_tokens=1024, temperature=0.1, stop=None):
    """Generate text from GGUF model. Returns the generated text."""
    if stop is None:
        stop = ["<|im_end|>", "<|endoftext|>"]
    
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.8,
        top_k=20,
        repeat_penalty=1.2,
        stop=stop,
    )
    return output["choices"][0]["text"]


def gguf_stream(prompt, max_tokens=1024, temperature=0.1, stop=None):
    """Stream text from GGUF model. Yields chunks."""
    if stop is None:
        stop = ["<|im_end|>", "<|endoftext|>"]
    
    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        repeat_penalty=1.2,
        stop=stop,
        stream=True,
    )
    for chunk in stream:
        token = chunk["choices"][0]["text"]
        if token:
            yield token

def truncate_prompt_to_fit(prompt, max_gen_tokens=1024):
    """Trim ONLY conversation history if prompt too long. NEVER touches tool results."""
    n_ctx = llm.n_ctx()
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    max_prompt_tokens = n_ctx - max_gen_tokens
    
    if prompt_tokens <= max_prompt_tokens:
        return prompt
    
    logger.warning(f"Prompt {prompt_tokens} tokens, limit {max_prompt_tokens}. Trimming history.")
    
    history_start = prompt.find("=== CONVERSATION HISTORY")
    history_end = prompt.find("=== END HISTORY ===")
    
    if history_start != -1 and history_end != -1:
        trimmed = prompt[:history_start] + prompt[history_end + len("=== END HISTORY ==="):]
        logger.info(f"Removed history. Now {len(llm.tokenize(trimmed.encode('utf-8')))} tokens.")
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
            Key sensor names: ME_RPM, SA_POW_act_kW@AVG (shaft power), ME_FMS_act_kgPh@AVG (ME fuel consumption), AE_FMS_act_kgPh@AVG (AE fuel), V_SOG_act_kn@AVG (speed), ME_Load@AVG (ME load %), ME_SCAV_AIR_PRESS (scav air pressure), ME_NO_1_TC_RPM (TC1 RPM), ME_NO_2_TC_RPM (TC2 RPM). VesselTimeStamp is unix epoch."""

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

        # Build prompt using tokenizer's chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tools=tools
        )

        # Generate with GGUF
        response = gguf_generate(text, max_tokens=1024, temperature=0.1)
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
                
                text2 = truncate_prompt_to_fit(text2, max_gen_tokens=1024)
                response = gguf_generate(text2, max_tokens=1024, temperature=0.1)

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

        #         response = gguf_generate(text2, max_tokens=1024, temperature=0.1)

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
    """Stream LLM response with thinking and answer separated"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        def generate():
            try:
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
                    
                    final_messages = [system_msg, current_user_msg]
                else:
                    final_messages = messages
                
                logger.info(f"=== STREAM PROMPT: {len(final_messages)} msgs, history={len(history_msgs)} msgs ===")

                # Build prompt with thinking enabled
                text = tokenizer.apply_chat_template(
                    final_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )

                # Stream from GGUF
                in_thinking = True
                buffer = ""

                for token in gguf_stream(text, max_tokens=1024, temperature=0.1):
                    buffer += token
                    
                    # Process buffer for think/answer transitions
                    if in_thinking and "</think>" in buffer:
                        parts = buffer.split("</think>", 1)
                        # Send remaining thinking content
                        think_content = parts[0].replace("<think>", "")
                        if think_content:
                            yield f"data: {json.dumps({'type': 'thinking', 'content': think_content})}\n\n"
                        in_thinking = False
                        buffer = parts[1] if len(parts) > 1 else ""
                        # Send any answer content after </think>
                        if buffer:
                            yield f"data: {json.dumps({'type': 'answer', 'content': buffer})}\n\n"
                            buffer = ""
                    elif in_thinking:
                        # Stream thinking tokens (but wait for enough to avoid partial tags)
                        safe = buffer.replace("<think>", "")
                        # Keep last 10 chars in buffer in case </think> is split across tokens
                        if len(safe) > 10:
                            send = safe[:-10]
                            buffer = "<think>" + safe[-10:] if "<think>" in buffer else safe[-10:]
                            if send:
                                yield f"data: {json.dumps({'type': 'thinking', 'content': send})}\n\n"
                    else:
                        # Answer mode — stream directly
                        clean = buffer.replace("<|im_end|>", "").replace("<|im_start|>", "")
                        if clean:
                            yield f"data: {json.dumps({'type': 'answer', 'content': clean})}\n\n"
                        buffer = ""
                
                # Flush remaining buffer
                if buffer:
                    clean = buffer.replace("<think>", "").replace("</think>", "").replace("<|im_end|>", "").replace("<|im_start|>", "")
                    if clean.strip():
                        msg_type = 'thinking' if in_thinking else 'answer'
                        yield f"data: {json.dumps({'type': msg_type, 'content': clean})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"Stream generation error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        logger.error(f"Error in LLM stream: {e}")
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
                'llm': llm is not None,
                'whisper': whisper_model is not None,
                'face_recognition': face_app is not None
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ============= STARTUP =============
if __name__ == '__main__':
    logger.info("Starting GPU Service (GGUF mode)...")
    app.run(host='0.0.0.0', port=5005, debug=False)