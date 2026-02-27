import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="insightface.utils.transform",
    message=".*rcond.*"
)
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging
import soundfile as sf
import io
import pickle
import pandas as pd
import json
from torch import nn
from pydub import AudioSegment
import numpy as np
from comtypes.client import CreateObject
from comtypes.gen import SpeechLib
from comtypes import CoInitialize, CoUninitialize
import tempfile
import base64
from scipy.signal import resample
import whisper
import requests 
import os
import sys
import contextlib 
import cv2 
from dotenv import load_dotenv
from tag_matcher1 import initialize_tag_matcher, enhanced_alarm_analysis
from test import initialize_fixed_processor, process_fixed_manual_query 

from werkzeug.utils import secure_filename
import tempfile
import shutil

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'
logger.info("Initializing manual processor...")
manual_processor = initialize_fixed_processor(db_path="./fixed_table_manual_db")
logger.info("Manual processor initialized successfully")

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

from insightface.app import FaceAnalysis

with suppress_stdout_stderr():
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    face_app.prepare(ctx_id=0)
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3' 

# Initialize Flask app
app = Flask('flk4')
app.url_map.strict_slashes = False

UPLOAD_FOLDER = './manual_uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- LLM SETUP ----------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SAVED_MODEL_PATH = r"C:\Users\User\Desktop\siemens\OFFSHORE\qwen_marine_maintenance_alpaca2"

# EXCEL_PATH = r"C:\Users\User\Desktop\siemens\OFFSHORE\Clemens tag with description.xlsx"  # Update this path
EXCEL_STORAGE_PATH = "./uploaded_excel/current_tags.xlsx"
os.makedirs("./uploaded_excel", exist_ok=True)
tag_matcher = None

logger.info("Tag matcher initialized successfully")

def load_ship_model(ship_id):
    """Load model and scalers for a specific ship"""
    # Fixed: Use the correct base path
    base_path = r'C:\Users\User\Desktop\siemens\clemens'
    model_dir = os.path.join(base_path, "model", ship_id)
    
    try:
        # Load hyperparameters
        params_path = os.path.join(model_dir, f'FULL{ship_id}_model_params.json')
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        
        # Define the neural network model with dynamic input size
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
        model = NeuralNetwork(
            input_size=model_params['pca_n_components'],
            n_layers=model_params['n_layers'],
            n_neurons=model_params['n_neurons'], 
            dropout_rate=model_params['dropout_rate'],
            l2_regularization=model_params['l2_regularization']
        )
        
        model_path = os.path.join(model_dir, f'FULL{ship_id}_pytorch_model.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
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
        
        return model, scaler_X, scaler_y, pca_X, device
        
    except Exception as e:
        raise FileNotFoundError(f"Model files not found for ship {ship_id}: {str(e)}")

def relative_wind_direction(wind_direction, ship_heading):
    relative_direction = wind_direction - ship_heading
    return np.mod(relative_direction + 360, 360)

@app.route('/excel/upload/', methods=['POST'])
def upload_excel():
    global tag_matcher
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({'error': 'Only Excel files allowed'}), 400
    
    try:
        # Save file (overwrite existing)
        file.save(EXCEL_STORAGE_PATH)
        
        # Recreate tag matcher (this will recreate ChromaDB)
        tag_matcher = initialize_tag_matcher(EXCEL_STORAGE_PATH)
        torch.cuda.empty_cache()
        
        return jsonify({'status': 'success', 'message': 'Excel uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/excel/delete/', methods=['DELETE'])
def delete_excel():
    global tag_matcher
    
    try:
        if os.path.exists(EXCEL_STORAGE_PATH):
            os.remove(EXCEL_STORAGE_PATH)
        
        if tag_matcher:
            tag_matcher.collection.delete()  # Clear ChromaDB
        
        tag_matcher = None
        torch.cuda.empty_cache()
        
        return jsonify({'status': 'success', 'message': 'Excel deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fuel/analysis', methods=['POST'])
def fuel_analysis():
    try:
        data = request.get_json()
        data = data['fuelmasterPayload']
        print(f"Raw input data: {data}")
        # Extract ship IMO
        imo_number = data.get('IMO')
        
        if not imo_number:
            return jsonify({'error': 'IMO is required'}), 400
        
        ship_id = f"IMO{imo_number}" 
        
        # Load ship-specific model and scalers
        try:
            model, scaler_X, scaler_y, pca_X, device = load_ship_model(ship_id)
            print("Model loaded successfully")
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        
        # Extract input values
        sog = data.get('V_SOG_act_kn@AVG')
        stw = data.get('V_STW_act_kn@AVG')
        rpm = data.get('SA_SPD_act_rpm@AVG')
        torque = data.get('SA_TQU_act_kNm@AVG')
        power = data.get('SA_POW_act_kW@AVG')
        actual_fuel = data.get('ME_FMS_act_kgPh@AVG')
        wind_direction = data.get('WEA_WDT_act_deg@AVG')
        ship_heading = data.get('V_HDG_act_deg@AVG')
        wind_speed = data.get('WEA_WST_act_kn@AVG')
        print(f"Extracted values - SOG: {sog}, STW: {stw}, RPM: {rpm}, Torque: {torque}")
        print(f"Power: {power}, Wind: {wind_speed}, WindDir: {wind_direction}, Heading: {ship_heading}")
        
        # Check for missing values
        required_fields = [sog, stw, rpm, torque, power, wind_direction, ship_heading, wind_speed]
        if any(x is None for x in required_fields):
            return jsonify({'error': 'Missing required input fields'}), 400
        
        # Calculate effective wind speed
        rel_wind_dir = relative_wind_direction(wind_direction, ship_heading)
        rel_wind_cos = np.cos(np.radians(rel_wind_dir))
        effective_wind_speed = wind_speed * rel_wind_cos
        print(f"Relative wind direction: {rel_wind_dir}")
        print(f"Effective wind speed: {effective_wind_speed}")
        # Prepare input data
        input_data = pd.DataFrame({
            'V_SOG_act_kn@AVG': [sog],
            'V_STW_act_kn@AVG': [stw], 
            'SA_SPD_act_rpm@AVG': [rpm],
            'SA_TQU_act_kNm@AVG': [torque],
            'Effective_Wind_Speed': [effective_wind_speed],
            'SA_POW_act_kW@AVG': [power]
        })
        print(f"Input DataFrame: {input_data}")
        # Preprocess
        X_scaled = scaler_X.transform(input_data)
        print(f"Scaled input: {X_scaled}")
        X_pca = pca_X.transform(X_scaled)
        print(f"PCA transformed: {X_pca}")
        X_tensor = torch.tensor(X_pca, dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            y_pred_scaled = model(X_tensor).cpu().numpy().flatten()
            print(f"Raw prediction (scaled): {y_pred_scaled}")
        predicted_fuel = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
        print(f"Final predicted fuel: {predicted_fuel}")
        
        # Custom logic for alert calculation
        alert = False
        
        if actual_fuel is not None and actual_fuel > 0:
            # Standard 4% tolerance check when we have valid actual fuel
            percentage_diff = abs(predicted_fuel - actual_fuel) / actual_fuel * 100
            alert = percentage_diff > 4.0
            print(f"Percentage diff: {percentage_diff}%, Alert: {alert}")
        else:
            # Custom logic when actual fuel is 0, None, or invalid
            # Alert if predicted fuel is unusually high (above a threshold)
            # You can adjust this threshold based on your domain knowledge
            fuel_threshold = 175  # Adjust this value based on your requirements
            alert = predicted_fuel > fuel_threshold
            print(f"Using threshold logic. Predicted: {predicted_fuel}, Threshold: {fuel_threshold}, Alert: {alert}")
        if alert == True:
            flag=1
        else:
            flag = 0
        final_result = {
        'predicted': round(float(predicted_fuel), 2),
        'alert': flag}
        print(f"Final result being returned: {final_result}")
        return jsonify(final_result)
        # return jsonify({
        #     'predicted': round(float(predicted_fuel), 2),
        #     'alert': alert
        # })
        
    except Exception as e:
        return jsonify({
            "error":str(e),
            'predicted': 0,
            'alert': True  # Always return alert=true for errors to be safe
        }), 200  # Return 200 to ensure frontend gets the response

@app.route('/manual/upload/', methods=['POST'])   
def upload_manual():
    """Upload and process a manual document"""
    if 'file' not in request.files:
        logger.warning("No file provided in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)
    
    if file_length > MAX_FILE_SIZE:
        logger.warning(f"File too large: {file_length} bytes")
        return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB'}), 400
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        file.save(temp_path)
        logger.info(f"Processing uploaded file: {filename}")
        
        # Process with manual processor
        result = manual_processor.process_document(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing manual upload: {e}")
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

logger.info("Loading tokenizer and Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
    quantization_config=bnb_config
)

model = PeftModel.from_pretrained(base_model, SAVED_MODEL_PATH)
model.eval()
# model=base_model
whisper_model = whisper.load_model("small").to("cuda")
logger.info("Qwen model loaded.")

# SYSTEM_PROMPT = """You are a senior marine engineering assistant. Respond only with what is asked—possible causes, corrective actions, or both. Do not over-explain. Use technical terms. Always end responses with a full stop. Avoid extra details. Be direct and precise."""
SYSTEM_PROMPT = """You are a senior marine engineering assistant. Keep responses concise and under 200 words. Be technical and direct. Always end with a complete sentence and full stop."""

# ALARM_SYSTEM_PROMPT = """You are a marine engineering expert. For alarm diagnostics:
# - List 5-7 most critical points only
# - Use bullet points or numbers
# - Maximum 20 words per point
# - No repetition of any phrase or sentence
# - Avoid similar or redundant points
# - Be direct and technical
# - Complete each sentence properly"""
ALARM_SYSTEM_PROMPT = """Marine engineering expert. Analyze alarms precisely.

Rules:
1. Maximum 5-7 points only
2. Each point under 20 words
3. Use numbered list (1. 2. 3.)
4. Be technical and direct
5. No repetition of any phrase or sentence

For POSSIBLE REASONS:
- State ONLY what failed/went wrong
- NO action words (check, verify, test, inspect)
- Good: "Power supply failure" 
- Bad: "Check power supply"

For CORRECTIVE ACTIONS:
- State ONLY fixing steps
- Use action verbs (check, replace, verify, test)
- Good: "Replace faulty sensor"
- Bad: "Sensor malfunction"

Stop after 7 points."""
def clean_response(text, response_type):
    """Remove unwanted content based on response type"""
    if response_type == "possible reasons":
        # Split into sentences
        sentences = text.split('.')
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not any(keyword in sentence.lower() for keyword in 
                ['corrective action', 'corrective measure', 'corrective solution', 'solution involve','corrective actions','corrective measures','corrective solutions','solutions involve']):
                cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences) + '.' if cleaned_sentences else text
    
    return text

@app.route('/manual/list/', methods=['GET'])
def list_manuals():
    """List all processed PDFs"""
    try:
        stats = manual_processor.get_stats()
        # Get unique document names from ChromaDB
        collection = manual_processor.vectorstore._collection
        data = collection.get()
        doc_names = set()
        for metadata in data.get('metadatas', []):
            if metadata and metadata.get('document_name'):
                doc_names.add(metadata['document_name'])
        
        return jsonify({
            'documents': list(doc_names),
            'total_chunks': stats.get('total_chunks', 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/manual/delete/<filename>', methods=['DELETE'])
def delete_manual(filename):
    """Delete specific PDF and its embeddings"""
    try:
        # Delete from ChromaDB by document_name filter
        collection = manual_processor.vectorstore._collection
        data = collection.get(where={"document_name": filename})
        
        if data['ids']:
            collection.delete(ids=data['ids'])
            torch.cuda.empty_cache()
            return jsonify({'status': 'success', 'message': f'{filename} deleted'})
        else:
            return jsonify({'error': 'Document not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------- ALARM ANALYSIS ENDPOINT ----------
@app.route('/alarm/analysis1/', methods=['POST'])                                   #################
def analyze_alarm():
    global tag_matcher
    if tag_matcher is None:
        return jsonify({'error': 'No excel uploaded. Upload vessel tags first.'}), 400
    data = request.get_json(force=True)
    # print(data)
    # alarm_name = (data.get('alarm_name') or data.get('alarm') or '').strip()
    alarm_list = data.get('alarm_name') or data.get('alarm') or []

    if not isinstance(alarm_list, list) or not alarm_list:
        logger.warning("No valid alarm list provided")
        return jsonify({'error': 'No valid alarm_name list provided.'}), 400
    
    response_data = []

    for alarm_name in alarm_list:
        alarm_name = str(alarm_name).strip()
        if not alarm_name:
            continue
        logger.info(f"Analyzing alarm: {alarm_name}")

    # Create prompts for possible reasons and corrective actions
        # reasons_prompt = f"What are the possible reasons for the alarm: {alarm_name}?Do NOT provide corrective actions or corrective measures!"
        # actions_prompt = f"What are the corrective actions for the alarm: {alarm_name}?"
        # For possible reasons
        reasons_prompt = f"Alarm: {alarm_name}. List ONLY what failed or went wrong. NO actions or fixes."

        # For corrective actions  
        actions_prompt = f"Alarm: {alarm_name}. List ONLY steps to fix the problem."
    
        logger.info(f"Analyzing alarm: {alarm_name}")
    
    # Get possible reasons
        reasons_messages = [
            {'role': 'system', 'content': ALARM_SYSTEM_PROMPT},
            {'role': 'user', 'content': reasons_prompt}
        ]
    
        reasons_result = process_fixed_manual_query(
        processor=manual_processor,
        question=reasons_prompt,
        llm_messages=reasons_messages,
        generate_llm_response_func=generate_llm_response
)
        # reasons_answer = clean_response(reasons_answer, "possible reasons")
    
    # Get corrective actions
        actions_messages = [
            {'role': 'system', 'content': ALARM_SYSTEM_PROMPT},
            {'role': 'user', 'content': actions_prompt}
        ]
    
        actions_result = process_fixed_manual_query(
        processor=manual_processor, 
        question=actions_prompt,
        llm_messages=actions_messages,
        generate_llm_response_func=generate_llm_response
    )
        reasons_answer = reasons_result['answer']
        actions_answer = actions_result['answer']

        enhanced_result = enhanced_alarm_analysis(
                tag_matcher=tag_matcher,
                alarm_name=alarm_name,
                possible_reasons=reasons_answer,
                corrective_actions=actions_answer,
                # alarm_tags=alarm_tags
            )
        response_data.append(enhanced_result)
    
    
    return jsonify({'data': response_data})

def generate_llm_response(messages, response_type):
    """Generate response using the LLM model"""
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    logger.info(f"Generating {response_type} with Qwen model")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=320,
            min_new_tokens=10,
            temperature=0.4,
            do_sample=True,
            top_k=45,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.2,
            
            # no_repeat_ngram_size=3,
            
            use_cache=True
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    assistant_start = response.find("assistant\n")
    if assistant_start != -1:
        response = response[assistant_start + len("assistant\n"):].strip()
    
    logger.info(f"Generated {response_type}: {response[:50]}...")
    return response

@app.route('/manual/query/', methods=['POST'])              ######/manual/query/
def query_manuals():
    """Query the manual database"""
    data = request.get_json(force=True)
    question = (data.get('question') or data.get('query') or '').strip()
    
    if not question:
        logger.warning("No question provided for manual query")
        return jsonify({'error': 'No question provided'}), 400
    
    logger.info(f"Manual query: {question}")
    
    try:
        # Build base messages
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question}
        ]
        
        # Process query with manual context
        result = process_fixed_manual_query(
        processor=manual_processor,
        question=question,
        llm_messages=messages,
        generate_llm_response_func=generate_llm_response
    )
        response = {
            'question': question,
            'answer': result['answer'],
            'source': result.get('source', 'unknown')  #added
        }

        # response['source'] = result.get('source', 'unknown')

        if result.get('source') == 'manual_context' and result.get('metadata'):
            all_pages = []
            for meta in result['metadata']:
                page = meta.get('page')  # Changed from 'pages' to 'page'
                if page is not None:
                    all_pages.append(page)
            # Remove duplicates and sort
            unique_pages = sorted(list(set(all_pages)))
            if unique_pages:
                response['pages'] = unique_pages

        # Add audio if requestedS
        if data.get('with_audio', False) and result.get('answer'):
            audio_blob = text_to_voice(result['answer'])
            result['audio_blob'] = audio_blob
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in manual query: {e}")
        return jsonify({'error': str(e)}), 500

# ---------- CHAT ENDPOINT (Modified to work without RAG) ----------

@app.route('/chat/response/', methods=['POST'])
def generate_response():
    data = request.get_json(force=True)
    question = (data.get('chat') or data.get('question') or '').strip()
    if not question:
        logger.warning("No question provided")
        return jsonify({'error': 'No question provided.'}), 400

    # Build prompt with RAG context
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question}
    ]

    result = process_fixed_manual_query(
        processor=manual_processor,
        question=question,
        llm_messages=messages,
        generate_llm_response_func=generate_llm_response
    )
    audio_blob = text_to_voice(result['answer'])

    return jsonify({'answer': result['answer'], 'blob': audio_blob})

def text_to_voice(text):
    logger.info("Converting text to speech")
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav_path = temp_wav.name
    temp_wav.close()

    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_mp3_path = temp_mp3.name
    temp_mp3.close()

    try:
        # Initialize Microsoft Speech Platform
        CoInitialize()
        engine = CreateObject("SAPI.SpVoice")
        stream = CreateObject("SAPI.SpFileStream")

        # Select English voice
        for voice in engine.GetVoices():
            if 'english' in voice.GetAttribute('Name').lower() or 'en_' in voice.GetAttribute('Name').lower():
                engine.Voice = voice
                break

        stream.Open(temp_wav_path, SpeechLib.SSFMCreateForWrite)
        engine.AudioOutputStream = stream
        engine.Rate = 1  # Approximate pyttsx3 rate of 160
        engine.Speak(text)
        stream.Close()

        audio = AudioSegment.from_wav(temp_wav_path)
        audio.export(temp_mp3_path, format="mp3")

        with open(temp_mp3_path, 'rb') as f:
            audio_data = f.read()
            audio_blob = base64.b64encode(audio_data).decode('utf-8')

    except Exception as e:
        logger.warning(f"Failed to convert text to speech: {str(e)}")
        return None
    finally:
        try:
            os.remove(temp_wav_path)
            os.remove(temp_mp3_path)
        except OSError as e:
            logger.warning(f"Failed to remove temporary files: {str(e)}")

    logger.info("Text-to-speech conversion complete")
    return audio_blob

@app.route('/audio/response/', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        logger.warning("No audio file provided")
        return jsonify({'error': 'No audio file provided.'}), 400

    audio_file = request.files['audio']
    
    try:
        audio_bytes = audio_file.read()
        if not audio_bytes:
            logger.warning("Empty audio file provided")
            return jsonify({'error': 'Empty audio file provided.'}), 400
        
        audio_buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(audio_buffer)
        
        if audio_data.size == 0:
            logger.warning("Invalid or empty audio data")
            return jsonify({'error': 'Invalid or empty audio data.'}), 400
        
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        if sample_rate != 16000:
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            if num_samples <= 0:
                logger.warning("Invalid audio length for resampling")
                return jsonify({'error': 'Invalid audio length for resampling.'}), 400
            audio_data = resample(audio_data, num_samples)
            
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1.0:
            audio_data /= max_abs
            
        logger.info("Transcribing audio with Whisper model")
        result = whisper_model.transcribe(audio_data, fp16=False, language="en", task="transcribe")
        transcription = result["text"]
        logger.info(f"Transcription: {transcription}")
        
        chat_resp_url = "http://localhost:5004/chat/response/"
        logger.info(f"Sending transcription to chat endpoint: {chat_resp_url}")
        resp = requests.post(chat_resp_url, json={"chat": transcription})
        dict1 = resp.json()
        dict1['transcription'] = transcription
        logger.info("Received response from chat endpoint")
        return jsonify(dict1)
    
    except sf.LibsndfileError as e:
        logger.error(f"Failed to read WAV file: {str(e)}")
        return jsonify({'error': f'Failed to read WAV file: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/audio/transcribe/', methods=['POST'])
def simple_transcription():
    if 'audio' not in request.files:
        logger.warning("No audio file provided for transcription")
        return jsonify({"success": False, "error": "No audio file provided."}), 400

    audio_file = request.files['audio']

    try:
        audio_bytes = audio_file.read()
        if not audio_bytes:
            logger.warning("Empty audio file provided for transcription")
            return jsonify({"success": False, "error": "Empty audio file."}), 400

        audio_buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(audio_buffer)

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 16000:
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            if num_samples <= 0:
                logger.warning("Invalid audio length for transcription")
                return jsonify({"success": False, "error": "Invalid audio length."}), 400
            audio_data = resample(audio_data, num_samples)

        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1.0:
            audio_data /= max_abs

        logger.info("Transcribing audio for simple transcription")
        result = whisper_model.transcribe(audio_data, fp16=False, language="en", task="transcribe")
        transcription = result["text"]
        logger.info(f"Transcription result: {transcription}")

        return jsonify({"success": True, "transcription": transcription})

    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/image/response/', methods=['POST'])
def compare_faces():
    if 'image' not in request.files or 'profilePicture' not in request.files:
        return jsonify({'error': 'Both image1 and image2 are required.'}), 400

    image_file1 = request.files['image']
    image_file2 = request.files['profilePicture']

    try:
        # Read the uploaded images using cv2.imdecode
        image1 = cv2.imdecode(np.frombuffer(image_file1.read(), np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(image_file2.read(), np.uint8), cv2.IMREAD_COLOR)

        if image1 is None:
            return jsonify({'error': 'No face found in image1.'}), 400
        if image2 is None:
            return jsonify({'error': 'No face found in image2.'}), 400

        # Get face embeddings using InsightFace
        faces1 = face_app.get(image1)
        faces2 = face_app.get(image2)

        if not faces1:
            return jsonify({'error': 'No face found in image1.'}), 400
        if not faces2:
            return jsonify({'error': 'No face found in image2.'}), 400

        # Use the first detected face
        encoding1 = faces1[0].embedding
        encoding2 = faces2[0].embedding

        # Normalize embeddings
        encoding1 = encoding1 / np.linalg.norm(encoding1)
        encoding2 = encoding2 / np.linalg.norm(encoding2)

        # Compute cosine similarity and scale to [0–1] range
        cosine_similarity = np.dot(encoding1, encoding2)  # Range: [-1, 1]
        similarity = (cosine_similarity + 1) / 2  # Scale to [0, 1]

        logger.info(f"Face similarity: {similarity:.4f}")
        if float(similarity) * 100 > 70:
            return jsonify({
                "success": True,
                "score": str(float(similarity) * 100)
            })
        else:
            return jsonify({
                "success": False
            })
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        print(str(e))
        return jsonify({'error': str(e)}), 500

# ---------- STARTUP ----------
if __name__ == '__main__':
    logger.info("Starting application...")
    PORT_ENV = os.getenv('PORT')
    from waitress import serve
    serve(app, host="0.0.0.0", port=5004)
    # serve(app, host="0.0.0.0", port=PORT_ENV)
    # app.run(host='0.0.0.0', port=PORT_ENV)