import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Hide GPU 1, only show GPU 0
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse,StreamingResponse
import httpx
import logging
from typing import List, Dict, Optional, Any
import traceback
import os
import tempfile
from pathlib import Path
import shutil
# import datetime
from datetime import datetime
import torch
import requests
from jit_calculator import run_jit_calculation
import json as json_lib
# Import all our components
from vessel_manager import VesselSpecificManager
from database import initialize_database, get_database_manager
from queue_manager import initialize_queue_manager, get_queue_manager, Priority, process_chat_immediately
from test3 import process_fixed_manual_query
# from faultsense import load_config_with_overrides, process_smart_maintenance_results, run_pipeline
import pickle
import pandas as pd
import json
from torch import nn
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Marine Engineering AI System",
    description="Vessel-specific marine engineering assistance with AI",
    version="2.0.0"
)

# Global managers
vessel_manager: VesselSpecificManager = None
db_manager = None
queue_manager = None

# ============= CONVERSATION MEMORY =============
from collections import OrderedDict
import time as _time

class ConversationMemory:
    """In-memory conversation history store, keyed by session_id.
    
    - Keeps last MAX_SESSIONS sessions (LRU eviction)
    - Each session stores up to MAX_TURNS user/assistant pairs
    - Auto-expires sessions after TTL_SECONDS of inactivity
    """
    MAX_SESSIONS = 200
    MAX_TURNS = 3         # 3 user+assistant pairs = 6 messages
    TTL_SECONDS = 3600    # 1 hour inactivity timeout

    def __init__(self):
        self._store: OrderedDict[str, dict] = OrderedDict()

    def _evict(self):
        """Remove expired and over-limit sessions."""
        now = _time.time()
        # Remove expired
        expired = [sid for sid, s in self._store.items() if now - s['last_access'] > self.TTL_SECONDS]
        for sid in expired:
            del self._store[sid]
        # Remove oldest if over limit
        while len(self._store) > self.MAX_SESSIONS:
            self._store.popitem(last=False)

    def get_history(self, session_id: str) -> list:
        """Return list of past messages [{role, content}, ...] for this session."""
        self._evict()
        if session_id in self._store:
            self._store[session_id]['last_access'] = _time.time()
            self._store.move_to_end(session_id)
            return list(self._store[session_id]['messages'])
        return []

    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str):
        """Store a user+assistant exchange."""
        self._evict()
        if session_id not in self._store:
            self._store[session_id] = {'messages': [], 'last_access': _time.time()}
        
        session = self._store[session_id]
        session['messages'].append({'role': 'user', 'content': user_msg})
        session['messages'].append({'role': 'assistant', 'content': assistant_msg})
        session['last_access'] = _time.time()
        
        # Trim to MAX_TURNS pairs (keep last N pairs)
        if len(session['messages']) > self.MAX_TURNS * 2:
            session['messages'] = session['messages'][-(self.MAX_TURNS * 2):]
        
        self._store.move_to_end(session_id)

    def clear_session(self, session_id: str):
        """Manually clear a session."""
        self._store.pop(session_id, None)

conversation_memory = ConversationMemory()

# File upload settings
UPLOAD_FOLDER = './temp_uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt'}
# MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_ship_model(ship_id):
    """Load model and scalers for a specific ship"""
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

def load_ae_model(ship_id):
    """Load AE model and scalers for a specific ship"""
    base_path = r'C:\Users\User\Desktop\siemens\clemens'
    model_dir = os.path.join(base_path, "AE_model", ship_id)
    
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

@app.on_event("startup")
async def startup_event():
    """Initialize all components at startup"""
    global vessel_manager, db_manager, queue_manager
    
    logger.info("Starting Marine Engineering AI System...")
    
    # Create upload folder
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize vessel manager
    vessel_manager = VesselSpecificManager()
    logger.info("Vessel manager initialized")
    
    # Initialize database
    db_manager = initialize_database()
    
    # Test database connection
    if db_manager.test_connection():
        logger.info("Database connection successful")
    else:
        logger.warning("Database connection failed - alarm caching disabled")
    
    # Initialize queue manager
    queue_manager = initialize_queue_manager("http://localhost:5005")
    logger.info("Queue manager initialized")
    
    logger.info("All systems initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Marine Engineering AI System...")
    
    # Cleanup temp files
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    
    logger.info("Shutdown complete")

# ============= VESSEL MANAGEMENT =============
@app.get("/vessels/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "vessel_manager": "initialized" if vessel_manager else "not_initialized",
        "database": "connected" if db_manager and db_manager.test_connection() else "disconnected",
        "queue_manager": "initialized" if queue_manager else "not_initialized",
        "queue_status": queue_manager.get_queue_status() if queue_manager else None
    }
# ============= EXCEL TAG MANAGEMENT =============

@app.post("/vessels/{imo}/tags/upload")
async def upload_vessel_tags(imo: str, file: UploadFile = File(...)):
    """Upload tags Excel for specific vessel"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files allowed")
        
        # Save to temp location
        temp_path = os.path.join(UPLOAD_FOLDER, f"tags_{imo}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get vessel instance and upload
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.upload_tags_excel(temp_path)
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error uploading tags for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/vessels/{imo}/tags")
async def delete_vessel_tags(imo: str):
    """Delete tags for specific vessel"""
    try:
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.delete_tags_excel()
        return result
        
    except Exception as e:
        logger.error(f"Error deleting tags for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= MANUAL MANAGEMENT =============


@app.post("/vessels/{imo}/manuals/upload/")
async def upload_vessel_manual(imo: str, file: UploadFile = File(...)):
    """Upload manual for specific vessel"""
    print("IMOM", imo)
    try:
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # Save to temp location
        temp_path = os.path.join(UPLOAD_FOLDER, f"manual_{imo}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Add manual upload to queue (low priority)
        # task_id = await queue_manager.add_task(
        #     task_type="manual_upload",
        #     endpoint="local_manual_processing",  # Special marker for local processing
        #     payload={"vessel_imo": imo, "file_path": temp_path},
        #     priority=Priority.MANUAL_UPLOAD,
        #     vessel_imo=imo
        # )
        # Queue task is just for logging, don't let it block upload
        try:
            task_id = await queue_manager.add_task(
                task_type="manual_upload",
                endpoint="local_manual_processing",
                payload={"vessel_imo": imo, "file_path": temp_path},
                priority=Priority.MANUAL_UPLOAD,
                vessel_imo=imo
            )
        except Exception as e:
            logger.warning(f"Queue task failed (non-blocking): {e}")

        logger.info(f"Starting manual upload for vessel {imo}")
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.upload_manual(temp_path)
        logger.info(f"Upload result: {result}")    
        
        # Process locally (not GPU operation)
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.upload_manual(temp_path)
        torch.cuda.empty_cache()
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error uploading manual for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vessels/{imo}/manuals")
async def list_vessel_manuals(imo: str):
    """List all manuals for specific vessel"""
    try:
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.list_manuals()
        return result
        
    except Exception as e:
        logger.error(f"Error listing manuals for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/vessels/{imo}/manuals/{filename}")
async def delete_vessel_manual(imo: str, filename: str):
    """Delete specific manual for vessel"""
    try:
        vessel = vessel_manager.get_vessel_instance(imo)
        result = vessel.delete_manual(filename)
        return result
        
    except Exception as e:
        logger.error(f"Error deleting manual for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= ALARM ANALYSIS =============

@app.post("/vessels/{imo}/alarms/analyze")
async def analyze_vessel_alarms(imo: str, request_data: Dict[str, Any]):
    """Analyze alarms for specific vessel with caching"""
    try:
        alarm_list = request_data.get('alarm_name') or request_data.get('alarm') or []
        
        if not isinstance(alarm_list, list) or not alarm_list:
            raise HTTPException(status_code=400, detail="No valid alarm_name list provided")
        
        vessel = vessel_manager.get_vessel_instance(imo)
        if not vessel.tag_matcher:
         raise HTTPException(status_code=400, detail=f'No tags uploaded for vessel {imo}. Please upload tags first.')
        response_data = []
        
        for alarm_name in alarm_list:
            alarm_name = str(alarm_name).strip()
            if not alarm_name:
                continue
            
            logger.info(f"Analyzing alarm for vessel {imo}: {alarm_name}")
            
            # Check cache first
            cached_result = db_manager.check_alarm_cache(imo, alarm_name)
            
            if cached_result:
                logger.info(f"Using cached result for vessel {imo}, alarm: {alarm_name}")
                response_data.append(cached_result)
                continue
            
            # Generate new analysis
            logger.info(f"Generating new analysis for vessel {imo}, alarm: {alarm_name}")
            
            # Create system prompt for alarm analysis
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

            # Generate possible reasons
            reasons_prompt = f"Alarm: {alarm_name}. List ONLY what failed or went wrong. NO actions or fixes."
            reasons_messages = [
                {'role': 'system', 'content': ALARM_SYSTEM_PROMPT},
                {'role': 'user', 'content': reasons_prompt}
            ]
            
            # Generate corrective actions
            actions_prompt = f"Alarm: {alarm_name}. List ONLY steps to fix the problem."
            actions_messages = [
                {'role': 'system', 'content': ALARM_SYSTEM_PROMPT},
                {'role': 'user', 'content': actions_prompt}
            ]
            
            # Process reasons (queued)
            def generate_llm_response_sync(messages, response_type):
                
                response = requests.post("http://localhost:5005/gpu/llm/generate", 
                                    json={"messages": messages, "response_type": response_type}, 
                                    timeout=120)
                return response.json().get('response', '')
            
            reasons_result = process_fixed_manual_query(
            processor=vessel.get_manual_processor(),
            question=reasons_prompt,
            llm_messages=reasons_messages,
            generate_llm_response_func=generate_llm_response_sync
            )
            
            # Process actions (queued)
            actions_result = process_fixed_manual_query(
            processor=vessel.get_manual_processor(),
            question=actions_prompt, 
            llm_messages=actions_messages,
            generate_llm_response_func=generate_llm_response_sync
            )
            # print(f"DEBUG - reasons_result: {reasons_result}")
            # print(f"DEBUG - actions_result: {actions_result}")
            
            # Wait for results
            reasons_answer = reasons_result['answer']
            actions_answer = actions_result['answer']
            # print(f"DEBUG - reasons_answer: '{reasons_answer}'")
            # print(f"DEBUG - actions_answer: '{actions_answer}'")
            reasons_metadata = reasons_result.get('metadata', [])
            actions_metadata = actions_result.get('metadata', [])
            
            if not reasons_result or 'error' in reasons_result:
                logger.error(f"Failed to generate reasons for alarm {alarm_name}")
                continue
                
            if not actions_result or 'error' in actions_result:
                logger.error(f"Failed to generate actions for alarm {alarm_name}")
                continue
            
            
            # Enhanced alarm analysis with vessel tags
            enhanced_result = vessel.analyze_alarm(alarm_name, reasons_answer, actions_answer)

            
            if 'error' not in enhanced_result:
                enhanced_result['possible_reasons'] = reasons_answer
                enhanced_result['corrective_actions'] = actions_answer
                enhanced_result['reasons_metadata'] = reasons_metadata
                enhanced_result['actions_metadata'] = actions_metadata
            else:
                # Don't add anything to error response
                pass
            
            # Store in cache
            db_manager.store_alarm_analysis(
                vessel_imo=imo,
                alarm_name=alarm_name,
                possible_reasons=reasons_answer,
                corrective_actions=actions_answer,
                suspected_tags=enhanced_result.get('suspected_tags', []),
                metadata={
                'analysis_type': 'ai_generated',
                'model_version': '2.0',
                'reasons_sources': reasons_metadata,  # Add this
                'actions_sources': actions_metadata   # Add this
                }
                # metadata={
                #     'analysis_type': 'ai_generated',
                #     'model_version': '2.0'
                # }
            )
            # print(f"DEBUG - final enhanced_result before append: {enhanced_result}")
            response_data.append(enhanced_result)
            # print(f"####################DEBUG - final response_data: {response_data}")
        return {'data': response_data}
        
    except Exception as e:
        # logger.error(f"Error analyzing alarms for vessel {imo}: {e}")
        # raise HTTPException(status_code=500, detail=str(e))
        
        logger.error(f"Error analyzing alarms for vessel {imo}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/chat/response/")
async def chat_general(request_data: Dict[str, Any]):
    """General chat endpoint - NO RAG, NO IMO — with conversation memory"""
    try:
        question = (request_data.get('chat') or request_data.get('question') or '').strip()
        session_id = (request_data.get('session_id') or '').strip()
        logger.info(f"DEBUG session_id received: '{session_id}', history: {conversation_memory.get_history(session_id)}")
        logger.info(f"DEBUG for C/R session_id received: '{session_id}', history: {conversation_memory.get_history(session_id)}")
        
        if not question:
            return JSONResponse(content={'error': 'No question provided.'}, status_code=400)
        
        logger.info(f"General chat request, session: {session_id or 'none'}, Q: {question}")
        
        # Build messages with history
        system_msg = {'role': 'system', 'content': 'You are a friendly and knowledgeable marine engineering assistant. Be helpful, conversational, and concise. Plain text only, no HTML.'}
        
        messages = [system_msg]
        
        # Inject conversation history if session_id provided
        if session_id:
            history = conversation_memory.get_history(session_id)
            if history:
                messages.extend(history)
                logger.info(f"Injected {len(history)} history messages for session {session_id}")
        
        messages.append({'role': 'user', 'content': question})
        
        response = requests.post(
            "http://localhost:5005/gpu/llm/generate",
            json={"messages": messages, "response_type": "general_chat","imo": request_data.get('imo')},
            timeout=200
        )
        answer = response.json().get('response', '')
        
        # Store exchange in memory
        if session_id and answer:
            conversation_memory.add_exchange(session_id, question, answer)
        
        data_object = {'answer': answer}
        
        # Generate audio
        if answer:
            audio_result = await queue_manager.process_immediately(
                task_type="text_to_speech",
                endpoint="/gpu/tts/generate",
                payload={"text": answer},
                vessel_imo=None
            )
            audio_blob = audio_result.get('audio_blob', '') if audio_result and not audio_result.get('error') else ''
            if audio_blob:
                data_object['blob'] = audio_blob
        
        return JSONResponse(content={'data': data_object})
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return JSONResponse(content={'error': str(e)}, status_code=500)
    

@app.post("/chat/stream/")
async def chat_stream(request_data: Dict[str, Any]):
    """Streaming chat endpoint - SSE — with conversation memory"""
    question = (request_data.get('chat') or request_data.get('question') or '').strip()
    vessel_imo = (request_data.get('imo') or '').strip()
    session_id = (request_data.get('session_id') or '').strip()
    enable_thinking = request_data.get('enable_thinking', False)
    logger.info(f"DEBUG ROUTING - vessel_imo: '{vessel_imo}', enable_thinking: {enable_thinking}, type: {type(enable_thinking)}")
    
    if not question:
        return JSONResponse(content={'error': 'No question provided.'}, status_code=400)
    
    logger.info(f"Stream request - IMO: {vessel_imo if vessel_imo else 'GENERAL'}, session: {session_id or 'none'}, Q: {question}")
    
    async def event_generator():

        full_answer = ""
        try:
            # If vessel-specific, get RAG context first
            context = ""
            metadata = []
            
            if vessel_imo and enable_thinking:
                vessel = vessel_manager.get_vessel_instance(vessel_imo)
                # Rewrite follow-up queries for better embedding search
                search_query = question
                logger.info(f"DEBUG STREAM RAG - vessel_imo: {vessel_imo}, session_id: {session_id}, original question: {question}")
                if session_id:
                    history = conversation_memory.get_history(session_id)
                    if history:
                        from test3 import rewrite_followup_query
                        def _llm_call(msgs, rt):
                            return requests.post(
                                "http://localhost:5005/gpu/llm/generate",
                                json={"messages": msgs, "response_type": rt},
                                timeout=200
                            ).json().get('response', '')
                        search_query = rewrite_followup_query(question, history, _llm_call)
                
                query_result = vessel.get_manual_processor().query_manuals(search_query, n_results=15)
                logger.info(f"DEBUG STREAM RAG - search_query going to embedding: '{search_query}'")
                if query_result.get('context'):
                    from test3 import _merge_overlapping_chunks
                    context = _merge_overlapping_chunks(
                        query_result['context'],
                        query_result.get('metadata_detailed', [])
                    )
                    metadata = query_result.get('metadata', [])
            logger.info(f"DEBUG CONTEXT - context_found: {bool(context)}, context_length: {len(context)}")
            
            # Build user content
            if context:
                user_content = f"""You are answering based on manual excerpts.

    RULES YOU MUST FOLLOW:
    1. If the context contains a NUMBERED PROCEDURE (Step 1, Step 2... or 1., 2., 3...), you MUST list EVERY step. Do NOT summarize steps. Do NOT skip any step.
    2. Multiple context sections may contain OVERLAPPING parts of the same procedure. Combine them into ONE complete list with ALL unique steps.
    3. Copy ALL numbers, codes, values EXACTLY as written. Never approximate.
    4. If information is not in the context, say so.
    5. DO NOT SUMMARIZE. LIST EVERY STEP IF IT IS A PROCEDURE.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER (list every step if it's a procedure):"""
            else:
                user_content = question
    #             user_content = f"""You are a friendly and knowledgeable marine engineering assistant. Answer in English only. Plain text only, no HTML tags or markdown. Never include special tokens or end-of-sentence markers in your response. Be concise and helpful.

    # {question}"""
            
            # Build messages WITH history
            messages = [
                {'role': 'system', 'content': 'You are a marine and offshore engineering expert. Provide accurate, concise, and technically sound answers.'},
            ]
            
            # Inject conversation history
            if session_id:
                history = conversation_memory.get_history(session_id)
                if history:
                    messages.extend(history)
                    logger.info(f"Injected {len(history)} history messages for stream session {session_id}")
            
            messages.append({'role': 'user', 'content': user_content})
            
            # ============================================================
            # DECISION: Which GPU endpoint to use?
            #
            # ALWAYS use /gpu/llm/stream-chat now:
            #   - With IMO: tools enabled, handles tool calls + streams answer
            #   - Without IMO: no tools, just streams answer directly
            #   - RAG context is already baked into user_content above
            #
            # For RAG + thinking mode, we still want thinking visible,
            # so we use /gpu/llm/stream for that case.
            # ============================================================
            
            use_thinking = bool(context)  # RAG = thinking mode, general/tools = no thinking
            
            if use_thinking:
                # RAG case: use /gpu/llm/stream for thinking + answer streaming
                gpu_url = 'http://localhost:5005/gpu/llm/stream'
                gpu_payload = {'messages': messages, 'imo': vessel_imo}
            else:
                # General chat OR tool-capable: use /gpu/llm/stream-chat
                gpu_url = 'http://localhost:5005/gpu/llm/stream-chat'
                gpu_payload = {
                    'messages': messages,
                    'imo': vessel_imo if vessel_imo else None,
                    'response_type': 'general_chat' if not vessel_imo else 'vessel_chat'
                }
            
            logger.info(f"STREAM routing to: {gpu_url} (thinking={use_thinking}, imo={vessel_imo})")
            logger.info(f"DEBUG FINAL ROUTE - gpu_url: {gpu_url}, payload_keys: {list(gpu_payload.keys())}")
            
            # Stream from GPU service
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream('POST', gpu_url, json=gpu_payload) as response:
                    async for line in response.aiter_lines():
                        if line.startswith('data:'):
                            try:
                                data = json.loads(line[5:].strip())
                                event_type = data.get('type', '')
                                
                                if event_type == 'answer':
                                    chunk = data.get('content', '')
                                    chunk = chunk.replace('<|im_end|>', '')
                                    chunk = chunk.replace('<|im_start|>', '')
                                    if chunk:
                                        data['content'] = chunk
                                        full_answer += chunk
                                        yield f"data: {json.dumps(data)}\n\n"
                                
                                elif event_type == 'thinking':
                                    # Forward thinking chunks (RAG mode)
                                    yield f"data: {json.dumps(data)}\n\n"
                                
                                elif event_type == 'tool_status':
                                    # Forward tool status ("Fetching latest vessel readings...")
                                    yield f"data: {json.dumps(data)}\n\n"
                                
                                elif event_type == 'done':
                                    # Don't yield done yet — we still need to send audio + metadata
                                    pass
                                
                                elif event_type == 'error':
                                    yield f"data: {json.dumps(data)}\n\n"
                                
                                else:
                                    yield f"{line}\n\n"
                            
                            except Exception as parse_error:
                                logger.warning(f"Failed to parse line: {parse_error}")
                                yield f"{line}\n\n"
                    
                    logger.info(f"STREAM COMPLETE - Full answer length: {len(full_answer)}")
                    
                    # Store exchange in memory AFTER stream completes
                    if session_id and full_answer.strip():
                        conversation_memory.add_exchange(session_id, question, full_answer.strip())
                        logger.info(f"Stored exchange in memory for session {session_id}")
                    
                    # Generate audio from cleaned answer
                    if full_answer.strip():
                        full_answer = full_answer.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()
                        
                        try:
                            audio_result = await queue_manager.process_immediately(
                                task_type="text_to_speech",
                                endpoint="/gpu/tts/generate",
                                payload={"text": full_answer},
                                vessel_imo=vessel_imo if vessel_imo else None
                            )
                            
                            if audio_result and audio_result.get('audio_blob'):
                                audio_blob = audio_result['audio_blob']
                                yield f"data: {json.dumps({'type': 'blob', 'content': audio_blob})}\n\n"
                                logger.info("Audio blob sent to frontend")
                        except Exception as audio_error:
                            logger.error(f"Audio generation failed: {audio_error}")
                    
                    # Send metadata if available
                    if metadata:
                        yield f"data: {json.dumps({'type': 'metadata', 'content': metadata})}\n\n"
                    
                    # Send done signal
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
        except Exception as e:
            logger.error(f"STREAM ERROR: {e}")
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        # return StreamingResponse(event_generator(), media_type="text/event-stream")
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

# ============= CONVERSATION MEMORY ENDPOINTS =============

@app.delete("/chat/memory/{session_id}")
async def clear_chat_memory(session_id: str):
    """Clear conversation memory for a session (call on 'New Chat')"""
    conversation_memory.clear_session(session_id)
    logger.info(f"Cleared memory for session {session_id}")
    return {"status": "cleared", "session_id": session_id}

@app.get("/chat/memory/{session_id}")
async def get_chat_memory(session_id: str):
    """View conversation memory for a session (debug)"""
    history = conversation_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history, "turn_count": len(history) // 2}

@app.post("/audio/transcribe/")
async def simple_transcription(audio: UploadFile = File(...)):
    """Audio transcription - old endpoint format"""
    try:
        result = await queue_manager.process_immediately(
            task_type="audio_transcription",
            endpoint="/gpu/stt/transcribe",
            payload={"files": {"audio": audio.file}},
            vessel_imo=None
        )
        
        if result and not result.get('error'):
            return JSONResponse(content={
                "success": True,
                "transcription": result.get('transcription', '')
            })
        else:
            return JSONResponse(
                content={
                    "success": False,
                    "error": result.get('error', 'Transcription failed')
                },
                status_code=500
            )
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

# @app.post("/convert/latex")
# async def convert_to_latex(request_data: Dict[str, Any]):
#     """Convert condition expressions to LaTeX format"""
#     try:
#         condition = (request_data.get('condition') or '').strip()
        
#         if not condition:
#             return JSONResponse(content={'error': 'No condition provided.'}, status_code=400)
        
#         logger.info(f"LaTeX conversion request: {condition}")
        
# #         system_prompt = r"""You are a LaTeX converter. Convert the given condition into LaTeX math notation.

# # Rules:
# # 1. Output ONLY the LaTeX expression. No explanation, no preamble, no markdown, no backticks.
# # 2. Wrap variable/sensor names in \text{}, e.g. \text{ME_RPM}
# # 3. Use standard LaTeX comparison operators: >, <, \geq, \leq, =, \neq
# # 4. Use \land for AND, \lor for OR
# # 5. Keep numbers as plain numbers
# # 6. If multiple conditions, combine them with \land or \lor as appropriate
# # 7. Do NOT wrap in $ signs or \[ \] delimiters
# # 8. Replace @ symbols in sensor names with underscores"""
#         SYSTEM_PROMPT = r"""Convert conditions to LaTeX. Output ONLY the LaTeX expression, nothing else.

#         Examples:
#         Input: if @ME_RPM is greater than 100 and @V_SOG is less than 5
#         Output: \text{ME_RPM} > 100 \land \text{V_SOG} < 5

#         Input: if @DG5_Power_kW is more than 8 or @ME_Torque_kNm equals zero
#         Output: \text{DG5_Power_kW} > 8 \lor \text{ME_Torque_kNm} = 0

#         Input: if @ME_LOAD is not equal to 50 and @AE_TEMP is at least 200
#         Output: \text{ME_LOAD} \neq 50 \land \text{AE_TEMP} \geq 200

#         Now convert:"""

#         messages = [
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': condition}
#         ]
        
#         response = requests.post(
#             "http://localhost:5005/gpu/llm/generate",
#             json={"messages": messages, "response_type": "latex_conversion"},
#             timeout=250
#         )
        
#         latex = response.json().get('response', '').strip()
        
#         # Clean any accidental wrapping
#         latex = latex.replace('```latex', '').replace('```', '').strip()
#         latex = latex.strip('$').strip()
        
#         return JSONResponse(content={'latex': latex})
        
#     except Exception as e:
#         logger.error(f"Error in LaTeX conversion: {e}")
#         return JSONResponse(content={'error': str(e)}, status_code=500)

from condition_resolver import resolve_condition

@app.post("/convert/latex")
async def convert_to_latex(request_data: Dict[str, Any]):
    """Convert condition expressions to LaTeX format.
    Handles both:
      - Tag-based: "if @ME_RPM > 100 and @V_SOG < 5"
      - Natural language: "alert me if average exhaust temp exceeds 400"
    """
    try:
        condition = (request_data.get('condition') or '').strip()
        
        if not condition:
            return JSONResponse(content={'error': 'No condition provided.'}, status_code=400)
        
        logger.info(f"Condition resolve request: {condition}")
        
        result = resolve_condition(
            condition,
            llm_url="http://localhost:5005/gpu/llm/generate",
            latex_url="http://localhost:5020/convert/latex"
        )
        
        if not result["success"]:
            return JSONResponse(content={'error': result.get('error', 'Conversion failed')}, status_code=400)
        
        response_data = {
            'latex': result['latex'],
            'input_type': result['input_type'],
            'tag_condition': result.get('tag_condition', ''),
        }
        
        # Include structured rule if natural language was resolved
        if result.get('rule'):
            response_data['rule'] = result['rule']
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error in condition resolution: {e}")
        return JSONResponse(content={'error': str(e)}, status_code=500)

# ============= JIT RECOMMENDATION =============

@app.post("/vessels/{imo}/jit/recommend")
async def jit_recommend(imo: str, request_data: Dict[str, Any]):
    """
    JIT Arrival Recommendation
    Input:  { "port_name": "Savannah", "etb": "2026-03-10T04:00:00Z" }
    Output: instruction card JSON
    """
    try:
        port_name = (request_data.get("port_name") or "").strip()
        etb_iso   = (request_data.get("etb") or "").strip()

        if not port_name:
            raise HTTPException(status_code=400, detail="port_name is required")
        if not etb_iso:
            raise HTTPException(status_code=400, detail="etb is required")

        # --- Load ports.json ---
        ports_path = os.path.join(os.path.dirname(__file__), "ports.json")
        with open(ports_path) as f:
            ports = json_lib.load(f)

        port = ports.get(port_name)
        if not port:
            # Try case-insensitive match
            port_name_lower = port_name.lower()
            for k, v in ports.items():
                if k.lower() == port_name_lower:
                    port = v
                    port_name = k
                    break

        if not port or port.get("lat", 0) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Port '{port_name}' not found in ports database"
            )

        # --- Get vessel snapshot from telemetry ---
        logger.info(f"JIT: Fetching snapshot for IMO {imo}")
        import requests as req
        snapshot_resp = req.post(
            "http://localhost:5010/mcp/call",
            json={"name": "get_jit_snapshot", "arguments": {"imo": imo}},
            timeout=15
        )
        snapshot = snapshot_resp.json()

        if "error" in snapshot:
            raise HTTPException(status_code=500, detail=f"Telemetry error: {snapshot['error']}")

        current = snapshot.get("current_state", {})
        sailing = snapshot.get("sailing_averages", {})

        lat = current.get("latitude")
        lon = current.get("longitude")

        if not lat or not lon:
            raise HTTPException(status_code=500, detail="Vessel position unavailable")

        # --- Run JIT math ---
        logger.info(f"JIT: Running calculation for {imo} → {port_name}")
        calc = run_jit_calculation(
            imo=imo,
            current_lat=lat,
            current_lon=lon,
            current_speed=current.get("sog_knots") or 0,
            current_fuel=current.get("fuel_kgph"),
            avg_speed=sailing.get("avg_sog_knots"),
            avg_fuel=sailing.get("avg_fuel_kgph"),
            destination_lat=port["lat"],
            destination_lon=port["lon"],
            etb_iso=etb_iso
        )

        if "error" in calc:
            raise HTTPException(status_code=400, detail=calc["error"])
        if calc.get("recommendation") == "IMPOSSIBLE":
            card = {
                "card_type":            "JIT_ARRIVAL",
                "vessel_imo":           imo,
                "destination_port":     port_name,
                "etb":                  etb_iso,
                "recommendation":       "IMPOSSIBLE",
                "required_speed_kn":    calc["required_speed_kn"],
                "max_speed_kn":         calc["max_speed_kn"],
                "distance_nm":          calc["distance_to_port_nm"],
                "hours_until_etb":      calc["hours_until_etb"],
                "berth_confidence_pct": calc["berth_confidence_pct"],
                "anchorage_risk_pct":   calc["anchorage_risk_pct"],
                "instruction_text":     calc["note"],
                "generated_at":         calc["calculated_at"],
                "status":               "PENDING"
            }
            return JSONResponse(content={"data": card})    


        # --- Generate instruction card via Qwen ---
        logger.info(f"JIT: Generating instruction card")

        SYSTEM_PROMPT = """You are a vessel operations AI generating JIT (Just-In-Time) arrival instruction cards for ship crew.
        Write in clear, professional maritime language. Be direct and concise.
        Plain text only. No markdown. No bullet points. No HTML."""
        fuel_abs = abs(calc['fuel_saving_tonnes'])
        fuel_usd_abs = abs(calc['fuel_saving_usd'])
        if calc['fuel_saving_tonnes'] > 0:
            fuel_sentence = f"Slowing down saves {fuel_abs} tonnes of fuel (approx. USD {fuel_usd_abs})."
        elif calc['fuel_saving_tonnes'] < 0:
            fuel_sentence = f"Speeding up costs an additional {fuel_abs} tonnes of fuel (approx. USD {fuel_usd_abs}), but ensures the vessel makes the berth window and avoids anchorage delays."
        else:
            fuel_sentence = "No change in fuel consumption expected."
        user_prompt = f"""Generate a JIT arrival instruction card based on this data:

Vessel IMO: {imo}
Destination Port: {port_name}
ETB (Berth Ready Time): {etb_iso}
Current Location: {current.get('location', 'At sea')}
Current Speed: {calc['current_speed_kn']} knots
Recommended Speed: {calc['required_speed_kn']} knots
Distance to Port: {calc['distance_to_port_nm']} nautical miles
Hours Until ETB: {calc['hours_until_etb']} hours
ETA at Current Speed: {calc['eta_at_current_speed_h']} hours from now
Early Arrival if No Change: {calc['early_by_hours']} hours early (0 means vessel is behind schedule)
Fuel at Current Speed: {calc['fuel_at_current_speed_kgph']} kg/h
Fuel at Recommended Speed: {calc['fuel_at_required_speed_kgph']} kg/h
Fuel impact: {fuel_sentence}
Recommendation: {calc['recommendation']}
Note: {'Vessel must go FASTER to make ETB. Extra fuel cost applies.' if calc['recommendation'] == 'SPEED_UP' else 'Vessel can SLOW DOWN and arrive just-in-time. Fuel savings apply.' if calc['recommendation'] == 'SLOW_DOWN' else 'Current speed is fine.'}
Berth Confidence: {calc['berth_confidence_pct']}%
Anchorage Risk if Ignored: {calc['anchorage_risk_pct']}% (this means: if the Master does NOT follow this recommendation, there is a {calc['anchorage_risk_pct']}% chance the vessel will have to anchor and wait for the berth window)

Write a short instruction card (4-6 sentences) telling the Master what to do, why, and what is saved. End with the confidence and anchorage risk."""
# Anchorage Risk if Ignored: {calc['anchorage_risk_pct']}%

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ]

        llm_resp = req.post(
            "http://localhost:5005/gpu/llm/generate",
            json={"messages": messages, "response_type": "jit_card"},
            timeout=120
        )
        card_text = llm_resp.json().get("response", "").strip()
        card_text = card_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

        # --- Build final card ---
        card = {
            "card_type":              "JIT_ARRIVAL",
            "vessel_imo":             imo,
            "destination_port":       port_name,
            "etb":                    etb_iso,
            "recommendation":         calc["recommendation"],
            "current_speed_kn":       calc["current_speed_kn"],
            "required_speed_kn":      calc["required_speed_kn"],
            "distance_nm":            calc["distance_to_port_nm"],
            "hours_until_etb":        calc["hours_until_etb"],
            "early_by_hours":         calc["early_by_hours"],
            "fuel_saving_tonnes":     calc["fuel_saving_tonnes"],
            "fuel_saving_usd":        calc["fuel_saving_usd"],
            "berth_confidence_pct":   calc["berth_confidence_pct"],
            "anchorage_risk_pct":     calc["anchorage_risk_pct"],
            "instruction_text":       card_text,
            "generated_at":           calc["calculated_at"],
            "status":                 "PENDING"
        }

        logger.info(f"JIT card generated for {imo} → {port_name}")
        return JSONResponse(content={"data": card})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JIT recommendation error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vessels/chat")  # Remove {imo} from path
async def chat_general(request_data: Dict[str, Any]):
    """Chat - checks for IMO in payload, falls back to general LLM"""
    try:
        question = (request_data.get('chat') or request_data.get('question') or '').strip()
        imo = request_data.get('imo') or request_data.get('IMO')  # Get IMO from body
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        logger.info(f"Chat request - IMO: {imo if imo else 'GENERAL'}, Q: {question}")
        
        SYSTEM_PROMPT = """You are a senior marine engineering assistant. Keep responses concise and under 200 words. Be technical and direct. Always end with a complete sentence and full stop."""
        
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question}
        ]
        
        # If IMO provided, use vessel context
        if imo:
            vessel = vessel_manager.get_vessel_instance(str(imo))
            
            result = process_fixed_manual_query(
                processor=vessel.get_manual_processor(),
                question=question,
                llm_messages=messages,
                generate_llm_response_func=lambda msgs, rt: requests.post(
                    "http://localhost:5005/gpu/llm/generate",
                    json={"messages": msgs, "response_type": rt},
                    timeout=200
                ).json().get('response', ''),
                conversation_history=conversation_memory.get_history(session_id) if session_id else None
            )
            result['vessel_imo'] = imo
        else:
            # General chat - no RAG, direct LLM
            response = requests.post(
                "http://localhost:5005/gpu/llm/generate",
                json={"messages": messages, "response_type": "general_chat"},
                timeout=200
            )
            result = {
                'question': question,
                'answer': response.json().get('response', ''),
                'source': 'llm_knowledge',
                'metadata': []
            }
        
        # Add audio
        if result.get('answer'):
            audio_result = await queue_manager.process_immediately(
                task_type="text_to_speech",
                endpoint="/gpu/tts/generate",
                payload={"text": result['answer']},
                vessel_imo=imo
            )
            if audio_result and not audio_result.get('error'):
                result['audio_blob'] = audio_result.get('audio_blob')
        
        return result
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))    

# ============= MANUAL QUERY =============

@app.post("/vessels/{imo}/manuals/query")
async def query_vessel_manuals(imo: str, request_data: Dict[str, Any]):
    """Query manuals for specific vessel"""
    try:
        question = (request_data.get('question') or request_data.get('query') or '').strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        logger.info(f"Manual query for vessel {imo}: {question}")
        
        vessel = vessel_manager.get_vessel_instance(imo)
        
        # Build messages for LLM
        SYSTEM_PROMPT = """You are a senior marine engineering assistant. Keep responses concise and under 200 words. Be technical and direct. Always end with a complete sentence and full stop."""
        
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question}
        ]
        
        # Add to medium priority queue
        task_id = await queue_manager.add_task(
            task_type="manual_query",
            endpoint="local_manual_query",  # Special marker for local processing
            payload={"vessel_imo": imo, "question": question, "messages": messages},
            priority=Priority.MANUAL_QUERY,
            vessel_imo=imo
        )
        
        # Process locally using vessel-specific manuals
        def generate_llm_response(messages, response_type):
            # This will be called by process_fixed_manual_query
            # We need to make a GPU service call
            # import requests
            try:
                response = requests.post(
                    "http://localhost:5005/gpu/llm/generate",
                    json={"messages": messages, "response_type": response_type},
                    timeout=200
                )
                response.raise_for_status()
                print(f"DEBUG - GPU service response: {result}")
                result = response.json()
                return result.get('response', '')
            except Exception as e:
                logger.error(f"GPU service call failed: {e}")
                return "I apologize, but I'm experiencing technical difficulties. Please try again later."
        
        # Process query with vessel-specific manual context
        result = process_fixed_manual_query(
            processor=vessel.get_manual_processor(),
            question=question,
            llm_messages=messages,
            generate_llm_response_func=generate_llm_response
        )
        
        # Add vessel info
        result['vessel_imo'] = imo
        
        return result
        
    except Exception as e:
        logger.error(f"Error querying manuals for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= CHAT OPERATIONS (HIGHEST PRIORITY) =============

@app.post("/vessels/{imo}/chat")
async def chat_with_vessel(imo: str, request_data: Dict[str, Any]):
    """Chat with AI using vessel-specific context (immediate processing)"""
    try:
        question = (request_data.get('chat') or request_data.get('question') or '').strip()
        # with_audio = request_data.get('with_audio', False)
        
        if not question:
            raise HTTPException(status_code=400, detail="No question provided")
        
        logger.info(f"Chat request for vessel {imo}: {question}")
        
        # Process chat query with vessel context
        vessel = vessel_manager.get_vessel_instance(imo)
        
        SYSTEM_PROMPT = """You are a senior marine engineering assistant. Keep responses concise and under 200 words. Be technical and direct. Always end with a complete sentence and full stop."""
        
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question}
        ]
        
        # Process locally using vessel-specific manuals
        def generate_llm_response_sync(messages, response_type):
            # Make synchronous call to GPU service for immediate processing
            import requests
            try:
                response = requests.post(
                    "http://localhost:5005/gpu/llm/generate",
                    json={"messages": messages, "response_type": response_type},
                    timeout=200
                )
                response.raise_for_status()
                result = response.json()
                return result.get('response', '')
            except Exception as e:
                logger.error(f"GPU service call failed: {e}")
                return "I apologize, but I'm experiencing technical difficulties. Please try again later."
        
        # Process with vessel-specific context
        result = process_fixed_manual_query(
            processor=vessel.get_manual_processor(),
            question=question,
            llm_messages=messages,
            generate_llm_response_func=generate_llm_response_sync
        )
        
        
        if result.get('answer'):
            # Process audio immediately, bypass queue
            audio_result = await queue_manager.process_immediately(
                task_type="text_to_speech",
                endpoint="/gpu/tts/generate", 
                payload={"text": result['answer']},
                vessel_imo=imo
            )
            
            if audio_result and not audio_result.get('error'):
                result['audio_blob'] = audio_result.get('audio_blob')
        
        # Add vessel info
        result['vessel_imo'] = imo
        
        return result
        
    except Exception as e:
        logger.error(f"Error in chat for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= AUDIO OPERATIONS =============

@app.post("/vessels/{imo}/audio/transcribe")
async def transcribe_audio(imo: str, audio: UploadFile = File(...)):
    """Transcribe audio for specific vessel"""
    try:
                      ##### CHANGED
        result = await queue_manager.process_immediately(
        task_type="audio_transcription",
        endpoint="/gpu/stt/transcribe",
        payload={"files": {"audio": audio.file}},
        vessel_imo=imo
    )
        
        if result and not result.get('error'):
            result['vessel_imo'] = imo
            
        return result
        
    except Exception as e:
        logger.error(f"Error transcribing audio for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vessels/{imo}/audio/chat")
async def audio_chat(imo: str, audio: UploadFile = File(...)):
    """Process audio chat for specific vessel"""
    try:
        
        transcribe_result = await queue_manager.process_immediately(
        task_type="audio_transcription",
        endpoint="/gpu/stt/transcribe", 
        payload={"files": {"audio": audio.file}},
        vessel_imo=imo
         )
        
        if not transcribe_result or transcribe_result.get('error'):
            raise HTTPException(status_code=500, detail="Audio transcription failed")
        
        transcription = transcribe_result.get('transcription', '')
        
        # Then process chat (immediate)
        chat_result = await chat_with_vessel(imo, {
            "chat": transcription,
            "with_audio": True
        })
        
        # Add transcription to result
        chat_result['transcription'] = transcription
        
        return chat_result
        
    except Exception as e:
        logger.error(f"Error in audio chat for vessel {imo}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= FACE RECOGNITION =============

# @app.post("/vessels/{imo}/face/compare")
@app.post("/image/response/")
async def compare_faces(
    # imo: str, 
    image: UploadFile = File(...), 
    profilePicture: UploadFile = File(...)
):
    """Compare faces for specific vessel"""
    try:
        # Add to high priority queue
        task_id = await queue_manager.add_task(
            task_type="face_comparison",
            endpoint="/gpu/face/compare",
            payload={"files": {"image": image.file, "profilePicture": profilePicture.file}},
            priority=Priority.FACE_RECOGNITION,
            # vessel_imo=imo
        )
        
        result = await queue_manager.get_task_result_async(task_id)
        
        if result and not result.get('error'):
            result.pop('vessel_imo',None)
            
        return result
        
    except Exception as e:
        logger.error(f"Error comparing faces : {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"Error comparing faces : {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= FUEL ANALYSIS =============
@app.post("/fuel/analysis")
async def fuel_analysis(request_data: Dict[str, Any]):
    """Analyze fuel consumption - handles both ME and AE"""
    try:
        # print(request_data)
        data = request_data
        
        # Handle different payload structures
        if 'fuelmasterPayload' in data:
            data = data['fuelmasterPayload']
        elif 'roundedPayloadAE' in data:
            data = data['roundedPayloadAE']
        else:
            raise HTTPException(status_code=400, detail='Invalid payload structure')
        
        # Extract ship IMO
        imo_number = data.get('IMO')
        
        if not imo_number:
            raise HTTPException(status_code=400, detail='IMO is required')
        
        ship_id = f"IMO{imo_number}" 
        
        # Try to load both models
        try:
            me_model, me_scaler_X, me_scaler_y, me_pca_X, device = load_ship_model(ship_id)
            me_available = True
        except FileNotFoundError:
            me_available = False

        try:
            ae_model, ae_scaler_X, ae_scaler_y, ae_pca_X, device = load_ae_model(ship_id)
            ae_available = True
        except FileNotFoundError:
            ae_available = False

        if not me_available and not ae_available:
            raise HTTPException(status_code=404, detail=f'No models found for {ship_id}')
        
        # Extract ME input values
        sog = data.get('V_SOG_act_kn@AVG')
        stw = data.get('V_STW_act_kn@AVG')
        rpm = data.get('SA_SPD_act_rpm@AVG')
        torque = data.get('SA_TQU_act_kNm@AVG')
        power = data.get('SA_POW_act_kW@AVG')
        actual_fuel = data.get('ME_FMS_act_kgPh@AVG')
        wind_direction = data.get('WEA_WDT_act_deg@AVG')
        ship_heading = data.get('V_HDG_act_deg@AVG')
        wind_speed = data.get('WEA_WST_act_kn@AVG')
        
        # Extract AE input values
        ae1_power = data.get('AE1_POW_act_kW@AVG')
        ae2_power = data.get('AE2_POW_act_kW@AVG')
        ae3_power = data.get('AE3_POW_act_kW@AVG')
        ae4_power = data.get('AE4_POW_act_kW@AVG')
        actual_ae_fuel = data.get('AE_HFO_FMS_act_kgPh@AVG')
        actual_ae_mdo_fuel = data.get('AE_MDO_FMS_act_kgPh@AVG')
        actual_ae_total_fuel = (actual_ae_fuel or 0) + (actual_ae_mdo_fuel or 0)
        
        # ME Prediction
        predicted_me = None
        alert_me = None
        
        required_fields = [sog, stw, rpm, torque, power, wind_direction, ship_heading, wind_speed]
        if me_available and all(x is not None for x in required_fields):
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
            X_scaled = me_scaler_X.transform(input_data)
            X_pca = me_pca_X.transform(X_scaled)
            X_tensor = torch.tensor(X_pca, dtype=torch.float32).to(device)
            
            # Predict
            with torch.no_grad():
                y_pred_scaled = me_model(X_tensor).cpu().numpy().flatten()
                
            predicted_fuel = me_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
            predicted_me = round(float(predicted_fuel), 2)
            
            if predicted_me < 0:
                predicted_me = 0
            
            
            
            alert_me = False
            if actual_fuel is not None and actual_fuel > 0:
                absolute_diff = abs(predicted_me - actual_fuel)
                
                # Simple: Only alert if difference is genuinely large
                if actual_fuel < 100:
                    alert_threshold = 100.0  # Need 100+ kg/h difference to care
                elif actual_fuel < 300:
                    alert_threshold = 150.0  # Need 150+ kg/h difference
                else:
                    alert_threshold = 200.0  # Need 200+ kg/h difference
                
                alert_me = absolute_diff > alert_threshold

        
        # AE Prediction
        predicted_ae = None
        alert_ae = None
        
        if ae_available and all(x is not None for x in [ae1_power, ae2_power, ae3_power, ae4_power]):
            ae_input = pd.DataFrame({
                'AE1_POW_act_kW@AVG': [ae1_power],
                'AE2_POW_act_kW@AVG': [ae2_power],
                'AE3_POW_act_kW@AVG': [ae3_power],
                'AE4_POW_act_kW@AVG': [ae4_power]
            })
            
            X_scaled = ae_scaler_X.transform(ae_input)
            X_pca = ae_pca_X.transform(X_scaled)
            X_tensor = torch.tensor(X_pca, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                y_pred_scaled = ae_model(X_tensor).cpu().numpy().flatten()
            
            predicted_ae = round(float(ae_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]), 2)
            
            if predicted_ae < 0:
                predicted_ae = 0
            
            # Alert logic
            alert_ae = False
            if actual_ae_total_fuel > 0:
                absolute_diff = abs(predicted_ae - actual_ae_total_fuel)
                percentage_diff = absolute_diff / actual_ae_total_fuel * 100
                
                # Alert if BOTH conditions met
                if percentage_diff > 8.0 and absolute_diff > 15:
                    alert_ae = True
        
        result = {
            'predicted_me': predicted_me,
            'predicted_ae': predicted_ae,
            'alert_ae': alert_ae,
            'alert_me': alert_me
        }
        
        # print("RESPP", result)
        return result
        
    except Exception as e:
        print("Error", e)
        return JSONResponse(
            content={
                "error": str(e),
                'predicted_me': None,
                'predicted_ae': None,
                'alert_me': None,
                'alert_ae': None
            },
            status_code=200
        )

# ============= STARTUP =============
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI Marine Engineering AI System...")
    uvicorn.run(app, host="0.0.0.0", port=5004)