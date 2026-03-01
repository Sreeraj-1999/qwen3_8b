"""
Merge LoRA adapter into base Qwen3-8B and save as full FP16 model.
Then convert to GGUF using llama.cpp.

Run this on your machine:
  python merge_and_convert.py

Requirements:
  pip install torch transformers peft accelerate --break-system-packages
  
After this script completes, you'll have:
  1. ./qwen3_marine_merged_fp16/  (full merged model — ~16GB, delete after GGUF)
  2. Then run llama.cpp to convert to GGUF (instructions printed at end)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === PATHS — CHANGE IF NEEDED ===
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = r"C:\Users\User\Desktop\siemens\OFFSHORE\qwen3_marine_final_v3"
MERGED_OUTPUT = r"C:\Users\User\Desktop\siemens\OFFSHORE\qwen3_marine_merged_fp16"

def main():
    # Step 1: Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    
    # Step 2: Load base model in FP16 (NO quantization — need full precision for merge)
    logger.info("Loading base model in FP16 (this uses ~16GB RAM, not GPU)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",          # Load on CPU — we have 128GB RAM
        trust_remote_code=True,
    )
    logger.info("Base model loaded on CPU")
    
    # Step 3: Load LoRA adapter
    logger.info(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, torch_dtype=torch.float16)
    logger.info("LoRA adapter loaded")
    
    # Step 4: Merge LoRA into base model
    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    logger.info("Merge complete!")
    
    # Step 5: Save merged model
    logger.info(f"Saving merged model to {MERGED_OUTPUT}...")
    os.makedirs(MERGED_OUTPUT, exist_ok=True)
    model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT)
    logger.info(f"Merged model saved to {MERGED_OUTPUT}")
    
    # Print size
    total_size = sum(
        os.path.getsize(os.path.join(MERGED_OUTPUT, f))
        for f in os.listdir(MERGED_OUTPUT)
        if os.path.isfile(os.path.join(MERGED_OUTPUT, f))
    )
    logger.info(f"Total size: {total_size / (1024**3):.1f} GB")
    
    # Step 6: Print next steps
    print("\n" + "=" * 60)
    print("MERGE COMPLETE! Next steps:")
    print("=" * 60)
    print(f"""
1. Clone llama.cpp (if not already):
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp

2. Install Python requirements:
   pip install -r requirements.txt

3. Convert to GGUF:
   python convert_hf_to_gguf.py "{MERGED_OUTPUT}" --outfile qwen3_marine_f16.gguf --outtype f16

4. Quantize to Q4_K_M:
   .\\build\\bin\\llama-quantize qwen3_marine_f16.gguf qwen3_marine_Q4_K_M.gguf Q4_K_M

   (If you haven't built llama.cpp yet:
    cmake -B build
    cmake --build build --config Release
   )

5. Your final file: qwen3_marine_Q4_K_M.gguf (~5GB)
   Delete the merged_fp16 folder and f16.gguf after this.

6. Test it:
   .\\build\\bin\\llama-cli -m qwen3_marine_Q4_K_M.gguf -p "What causes low lube oil pressure?" -n 256
""")

if __name__ == "__main__":
    main()