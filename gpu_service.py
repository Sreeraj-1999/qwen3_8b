import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.utils.transform", message=".*rcond.*")

# Clear comtypes cache BEFORE importing comtypes
import shutil
import tempfile
import os

cache_dir = os.path.join(tempfile.gettempdir(), "comtypes_cache")
if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        print(f"Cleared comtypes cache: {cache_dir}")
    except Exception as e:
        print(f"Could not clear cache: {e}")

# Now import everything else
from flask import Flask, request, jsonify,Response
import queue
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from mcp_tool_handler import TELEMETRY_TOOLS, execute_tool_call, parse_tool_call, needs_tool_call
from peft import PeftModel
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

logger.info("Loading deepseek model for LLM operations...")

# LLM MODEL SETUP
MODEL_NAME = "Qwen/Qwen3-8B"
SAVED_MODEL_PATH = r"C:\Users\User\Desktop\siemens\OFFSHORE\qwen3_marine_final_v3"


tokenizer = AutoTokenizer.from_pretrained(
    SAVED_MODEL_PATH, 
    trust_remote_code=True,
)
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype=torch.float16
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # torch_dtype=torch.float16,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
    quantization_config=bnb_config,
    local_files_only=True
)

model = PeftModel.from_pretrained(base_model, SAVED_MODEL_PATH)
model.eval()

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


# @app.route('/gpu/llm/generate', methods=['POST'])
# def generate_llm_response():
#     """Generate LLM response using DeepSeek model"""
#     try:
#         data = request.get_json()
#         messages = data.get('messages', [])
#         response_type = data.get('response_type', 'general')
#         enable_thinking = data.get('enable_thinking', False)  # non-thinking by default for /generate

#         if not messages:
#             return jsonify({'error': 'No messages provided'}), 400

#         # Build prompt using Qwen3 chat template
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=enable_thinking
#         )

#         inputs = tokenizer(text, return_tensors="pt").to(model.device)

#         logger.info(f"Generating {response_type} response with Qwen3 model")

#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=1024,
#                 temperature=0.7,
#                 top_p=0.8,
#                 top_k=20,
#                 do_sample=True,
#                 repetition_penalty=1.2,
#             )

#         response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

#         # Extract answer (after </think> if present)
#         if "</think>" in response:
#             response = response.split("</think>")[-1]

#         # Clean tokens
#         response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

#         logger.info(f"Generated {response_type}: {response[:80]}...")

#         return jsonify({
#             'response': response,
#             'response_type': response_type,
#             'status': 'success'
#         })
        
#     except Exception as e:
#         logger.error(f"Error in LLM generation: {e}")
#         return jsonify({'error': str(e)}), 500
#     finally:
#         torch.cuda.empty_cache()

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

        # Pass tools if IMO is available (vessel-specific context)
        tools = TELEMETRY_TOOLS if imo else None

        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tools=tools
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024,
                temperature=0.7, top_p=0.8, top_k=20,
                do_sample=True, repetition_penalty=1.2,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

        # Tool call loop (max 1 round)
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
                messages.append({"role": "user", "content": f"Tool result:\n{tool_result}\n\nAnswer the original question based on this data."})

                text2 = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False
                )
                inputs2 = tokenizer(text2, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs2 = model.generate(
                        **inputs2, max_new_tokens=512,
                        temperature=0.7, top_p=0.8, top_k=20,
                        do_sample=True, repetition_penalty=1.2,
                    )
                response = tokenizer.decode(outputs2[0][inputs2['input_ids'].shape[1]:], skip_special_tokens=False)

        # Clean
        if "</think>" in response:
            response = response.split("</think>")[-1]
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

        return jsonify({'response': response, 'response_type': response_type, 'status': 'success'})

    except Exception as e:
        logger.error(f"Error in LLM generation: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        torch.cuda.empty_cache()


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
                from transformers import TextIteratorStreamer
                
                # --- Same prompt building as /generate ---

                # Qwen3 chat template — thinking enabled for streaming
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )

                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.5,
                    top_p=0.95,
                    top_k=20,
                    do_sample=True,
                    repetition_penalty=1.2,
                    streamer=streamer
                )

                thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                in_thinking = True

                for new_text in streamer:
                    new_text = new_text.replace("<think>", "").replace("<|im_start|>", "")
                    if "<|im_end|>" in new_text:
                        clean_text = new_text.replace("<|im_end|>", "")
                        if clean_text.strip():
                            msg_type = 'thinking' if in_thinking else 'answer'
                            yield f"data: {json.dumps({'type': msg_type, 'content': clean_text})}\n\n"
                        break

                    if "</think>" in new_text:
                        parts = new_text.split("</think>")
                        if parts[0]:
                            yield f"data: {json.dumps({'type': 'thinking', 'content': parts[0]})}\n\n"
                        in_thinking = False
                        if len(parts) > 1 and parts[1]:
                            yield f"data: {json.dumps({'type': 'answer', 'content': parts[1]})}\n\n"
                    elif in_thinking:
                        if new_text:
                            yield f"data: {json.dumps({'type': 'thinking', 'content': new_text})}\n\n"
                    else:
                        if new_text:
                            yield f"data: {json.dumps({'type': 'answer', 'content': new_text})}\n\n"

                thread.join()
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            finally:
                torch.cuda.empty_cache()
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        logger.error(f"Error in LLM stream: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/gpu/stt/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Whisper model"""
    try:
        # if 'audio' not in request.files:
        #     return jsonify({'error': 'No audio file provided'}), 400     changed for app
        
        # audio_file = request.files['audio']
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
            
            # CRITICAL FIX: Get the proper gen directory path
            import comtypes.gen
            import comtypes
            
            # Try to get the directory path properly
            try:
                if hasattr(comtypes.gen, '__file__') and comtypes.gen.__file__ is not None:
                    gen_dir = os.path.dirname(comtypes.gen.__file__)
                else:
                    # Fallback: construct path from comtypes location
                    comtypes_path = os.path.dirname(comtypes.__file__)
                    gen_dir = os.path.join(comtypes_path, 'gen')
                
                # Ensure directory exists
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
            
            # Now generate the typelib
            from comtypes.client import GetModule
            GetModule("C:\\Windows\\System32\\Speech\\Common\\sapi.dll")
            
            # Import the generated library
            from comtypes.gen import SpeechLib
            
            # Create COM objects
            engine = CreateObject("SAPI.SpVoice")
            stream = CreateObject("SAPI.SpFileStream")

            # Select English voice
            voices = engine.GetVoices()
            for i in range(voices.Count):
                voice = voices.Item(i)
                voice_name = voice.GetAttribute('Name').lower()
                if 'english' in voice_name or 'en_' in voice_name or 'david' in voice_name or 'zira' in voice_name:
                    engine.Voice = voice
                    logger.info(f"Selected voice: {voice_name}")
                    break

            # Open stream and generate speech
            stream.Open(temp_wav_path, SpeechLib.SSFMCreateForWrite)
            engine.AudioOutputStream = stream
            engine.Rate = 1
            engine.Speak(text)
            stream.Close()

            # Convert WAV to MP3
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(temp_wav_path)
            audio.export(temp_mp3_path, format="mp3")

            # Read and encode MP3
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
            # Cleanup temp files
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

        # Read images
        image1 = cv2.imdecode(np.frombuffer(image_file1.read(), np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(image_file2.read(), np.uint8), cv2.IMREAD_COLOR)

        if image1 is None or image2 is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Get face embeddings
        faces1 = face_app.get(image1)
        faces2 = face_app.get(image2)

        if not faces1 or not faces2:
            return jsonify({'error': 'No face detected'}), 400

        # Use first detected face
        encoding1 = faces1[0].embedding
        encoding2 = faces2[0].embedding

        # Normalize embeddings
        encoding1 = encoding1 / np.linalg.norm(encoding1)
        encoding2 = encoding2 / np.linalg.norm(encoding2)

        # Compute similarity
        cosine_similarity = np.dot(encoding1, encoding2)
        similarity = (cosine_similarity + 1) / 2  # Scale to [0, 1]

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

        # Extract ship IMO
        imo_number = fuel_payload.get('IMO')
        if not imo_number:
            return jsonify({'error': 'IMO is required'}), 400
        
        ship_id = f"IMO{imo_number}" 
        
        # Load ship-specific model
        try:
            fuel_model, scaler_X, scaler_y, pca_X, device = load_ship_model(ship_id)
            logger.info("Fuel model loaded successfully")
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        
        # Extract input values
        sog = fuel_payload.get('V_SOG_act_kn@AVG')
        stw = fuel_payload.get('V_STW_act_kn@AVG')
        rpm = fuel_payload.get('SA_SPD_act_rpm@AVG')
        torque = fuel_payload.get('SA_TQU_act_kNm@AVG')
        power = fuel_payload.get('SA_POW_act_kW@AVG')
        actual_fuel = fuel_payload.get('ME_FMS_act_kgPh@AVG')
        wind_direction = fuel_payload.get('WEA_WDT_act_deg@AVG')
        ship_heading = fuel_payload.get('V_HDG_act_deg@AVG')
        wind_speed = fuel_payload.get('WEA_WST_act_kn@AVG')
        
        # Check for missing values
        required_fields = [sog, stw, rpm, torque, power, wind_direction, ship_heading, wind_speed]
        if any(x is None for x in required_fields):
            return jsonify({'error': 'Missing required input fields'}), 400
        
        # Calculate effective wind speed
        rel_wind_dir = relative_wind_direction(wind_direction, ship_heading)
        rel_wind_cos = np.cos(np.radians(rel_wind_dir))
        effective_wind_speed = wind_speed * rel_wind_cos
        
        # Prepare input data
        input_data = pd.DataFrame({
            'V_SOG_act_kn@AVG': [sog],
            'V_STW_act_kn@AVG': [stw], 
            'SA_SPD_act_rpm@AVG': [rpm],
            'SA_TQU_act_kNm@AVG': [torque],
            'Effective_Wind_Speed': [effective_wind_speed],
            'SA_POW_act_kW@AVG': [power]
        })
        
        # Preprocess
        X_scaled = scaler_X.transform(input_data)
        X_pca = pca_X.transform(X_scaled)
        X_tensor = torch.tensor(X_pca, dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            y_pred_scaled = fuel_model(X_tensor).cpu().numpy().flatten()
        predicted_fuel = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
        
        # Alert logic
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
                'llm': model is not None,
                'whisper': whisper_model is not None,
                'face_recognition': face_app is not None
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ============= STARTUP =============
if __name__ == '__main__':
    logger.info("Starting GPU Service...")
    # Use a different port than main FastAPI app
    app.run(host='0.0.0.0', port=5005, debug=False)