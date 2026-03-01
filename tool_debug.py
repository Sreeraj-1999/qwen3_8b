import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")

from llama_cpp import Llama
import torch
from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify, Response
from mcp_tool_handler import TELEMETRY_TOOLS, PMS_TOOLS, ALL_TOOLS, execute_tool_call, parse_tool_call, needs_tool_call
from transformers import AutoTokenizer
from embedding_service import embedding_service

print("Test: no split_mode...")
llm = Llama(
    model_path=r"C:\Users\User\Desktop\siemens\PROJECT5D\test\qwen3_marine_Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=99,
    main_gpu=0,
    cache_type_k="q8_0",
    cache_type_v="q8_0",
    verbose=False,
)
print("SUCCESS")
print("Generating...")
output = llm("What causes low lube oil pressure?", max_tokens=50)
print(output["choices"][0]["text"])