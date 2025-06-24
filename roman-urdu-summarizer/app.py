from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import hf_hub_download
import torch
import time

app = FastAPI()
model = None
tokenizer = None

class InputText(BaseModel):
    text: str

@app.on_event("startup")
def load_model_and_tokenizer():
    global model, tokenizer
    print("🔄 Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("radientsoul88/roman-urdu-summarizer")
    
    # Try to load PyTorch model first (much faster for generation)
    try:
        print("🔄 Attempting to load PyTorch model...")
        model = T5ForConditionalGeneration.from_pretrained("radientsoul88/roman-urdu-summarizer")
        
        # Set to evaluation mode
        model.eval()
        
        # Move to CPU (or GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ PyTorch model loaded on {device}")
        
        # Warm-up
        print("🔥 Running warm-up...")
        dummy_input = "summarize: test message"
        inputs = tokenizer(dummy_input, return_tensors="pt", max_length=64, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=20,
                num_beams=2,
                early_stopping=True,
                do_sample=False
            )
        print("🚀 Warm-up completed successfully.")
        
    except Exception as e:
        print(f"⚠️ Failed to load PyTorch model: {e}")
        print("🔄 Falling back to ONNX with optimized generation...")
        
        # Fallback to optimized ONNX approach
        model_path = hf_hub_download("radientsoul88/roman-urdu-summarizer", "t5_urdu_quant.onnx")
        
        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1  # Single thread often faster for small models
        session_options.intra_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        model = ort.InferenceSession(
            model_path, 
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )
        print("✅ ONNX model loaded with optimizations")

def generate_with_pytorch(input_text: str):
    """Fast generation using PyTorch model"""
    device = next(model.parameters()).device
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True, 
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=30,  # Adjust based on your needs
            min_length=5,
            num_beams=2,    # Beam search for better quality
            early_stopping=True,
            do_sample=False,  # Deterministic generation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=1.0
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()

def generate_with_onnx_optimized(input_text: str):
    """Optimized ONNX generation - use all outputs at once"""
    inputs = tokenizer(
        input_text, 
        return_tensors="np", 
        max_length=128, 
        truncation=True, 
        padding=True
    )
    
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    
    # Generate longer sequence at once instead of token by token
    decoder_input_ids = np.array([[tokenizer.pad_token_id] * 20], dtype=np.int64)  # Pre-allocate
    
    try:
        # Single inference call with longer decoder sequence
        outputs = model.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
            }
        )
        
        logits = outputs[0]
        # Take the most likely tokens across the sequence
        predicted_ids = np.argmax(logits, axis=-1)
        
        # Find EOS token position
        eos_positions = np.where(predicted_ids[0] == tokenizer.eos_token_id)[0]
        if len(eos_positions) > 0:
            predicted_ids = predicted_ids[0][:eos_positions[0]]
        else:
            predicted_ids = predicted_ids[0][:10]  # Take first 10 tokens
        
        summary = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        return summary.strip()
        
    except Exception as e:
        print(f"ONNX generation failed: {e}")
        return "Error in generation"

@app.post("/summarize")
async def summarize(input: InputText):
    start_time = time.time()
    
    input_text = f"summarize: {input.text}"
    
    try:
        if isinstance(model, T5ForConditionalGeneration):
            # Use PyTorch model (much faster)
            summary = generate_with_pytorch(input_text)
        else:
            # Use optimized ONNX approach
            summary = generate_with_onnx_optimized(input_text)
        
        generation_time = time.time() - start_time
        
        return {
            "summary": summary,
            "generation_time_seconds": round(generation_time, 2)
        }
        
    except Exception as e:
        return {
            "error": f"Summarization failed: {str(e)}",
            "generation_time_seconds": round(time.time() - start_time, 2)
        }

@app.get("/")
def root():
    return {"message": "Roman Urdu Summarizer API is running"}

@app.get("/health")
def health_check():
    model_type = "PyTorch" if isinstance(model, T5ForConditionalGeneration) else "ONNX"
    return {
        "status": "healthy",
        "model_type": model_type,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }
