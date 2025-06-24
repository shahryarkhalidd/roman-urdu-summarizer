from fastapi import FastAPI
from pydantic import BaseModel
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

app = FastAPI()

# Global variables for models
model_engine = None
tokenizer = None
executor = ThreadPoolExecutor(max_workers=1)

class InputText(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    global model_engine, tokenizer
    print("🔄 Loading lightweight model for fast deployment...")
    
    try:
        # First try the lightweight transformers approach (CPU-only)
        print("🔄 Loading PyTorch model (CPU-only)...")
        
        # Set environment variables to force CPU and avoid CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch
        
        # Force CPU usage
        torch.set_num_threads(2)  # Limit threads for faster startup
        
        print("📥 Downloading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained("radientsoul88/roman-urdu-summarizer")
        
        print("📥 Downloading model...")
        model_engine = T5ForConditionalGeneration.from_pretrained(
            "radientsoul88/roman-urdu-summarizer",
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model_engine.eval()
        
        print("✅ PyTorch model loaded successfully on CPU")
        
        # Quick warm-up
        print("🔥 Running quick warm-up...")
        test_input = tokenizer("summarize: test", return_tensors="pt", max_length=32, truncation=True)
        
        with torch.no_grad():
            outputs = model_engine.generate(
                **test_input,
                max_length=20,
                num_beams=1,  # Faster generation
                do_sample=False,
                early_stopping=True
            )
        
        test_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🚀 Warm-up completed. Test: {test_summary}")
        
    except Exception as e:
        print(f"⚠️ PyTorch loading failed: {e}")
        print("🔄 Trying lightweight ONNX approach...")
        
        try:
            # Fallback to basic ONNX with optimizations
            import onnxruntime as ort
            import numpy as np
            from transformers import T5Tokenizer
            from huggingface_hub import hf_hub_download
            
            print("📥 Loading ONNX model...")
            model_path = hf_hub_download("radientsoul88/roman-urdu-summarizer", "t5_urdu_quant.onnx")
            tokenizer = T5Tokenizer.from_pretrained("radientsoul88/roman-urdu-summarizer")
            
            # Optimized ONNX session for CPU
            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            model_engine = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"]
            )
            
            print("✅ ONNX model loaded successfully")
            
        except Exception as e2:
            print(f"❌ All model loading failed: {e2}")
            # Set a flag to indicate model loading failed
            model_engine = "failed"

def generate_summary_pytorch(input_text: str) -> str:
    """Fast PyTorch generation"""
    try:
        inputs = tokenizer(
            f"summarize: {input_text}",
            return_tensors="pt",
            max_length=96,  # Reduced for speed
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model_engine.generate(
                **inputs,
                max_length=25,  # Short summaries
                min_length=3,
                num_beams=1,    # Faster than beam search
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
        
    except Exception as e:
        return f"Generation failed: {str(e)}"

def generate_summary_onnx(input_text: str) -> str:
    """Optimized ONNX generation"""
    try:
        inputs = tokenizer(
            f"summarize: {input_text}",
            return_tensors="np",
            max_length=96,
            truncation=True,
            padding=True
        )
        
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
        
        # Generate up to 8 tokens (for speed)
        for _ in range(8):
            outputs = model_engine.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                }
            )
            
            logits = outputs[0]
            next_token_logits = logits[:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1, keepdims=True)
            
            if next_token_id[0, 0] == tokenizer.eos_token_id:
                break
                
            decoder_input_ids = np.hstack([decoder_input_ids, next_token_id])
        
        summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
        return summary.strip()
        
    except Exception as e:
        return f"ONNX generation failed: {str(e)}"

def generate_summary_sync(input_text: str) -> str:
    """Synchronous generation function"""
    if model_engine == "failed":
        return "Model failed to load"
    
    # Check model type and use appropriate generation
    if hasattr(model_engine, 'generate'):
        return generate_summary_pytorch(input_text)
    else:
        return generate_summary_onnx(input_text)

@app.post("/summarize")
async def summarize(input: InputText):
    start_time = time.time()
    
    try:
        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(executor, generate_summary_sync, input.text)
        
        generation_time = time.time() - start_time
        
        return {
            "summary": summary,
            "generation_time_seconds": round(generation_time, 2),
            "model_type": "PyTorch" if hasattr(model_engine, 'generate') else "ONNX"
        }
        
    except Exception as e:
        return {
            "error": f"Summarization failed: {str(e)}",
            "generation_time_seconds": round(time.time() - start_time, 2)
        }

@app.get("/")
def root():
    return {
        "message": "Roman Urdu Summarizer API is running",
        "status": "ready" if model_engine and model_engine != "failed" else "model_loading_failed"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_engine is not None and model_engine != "failed",
        "model_type": "PyTorch" if hasattr(model_engine, 'generate') else "ONNX" if model_engine else "failed"
    }

# Simple test endpoint
@app.get("/test")
def test_endpoint():
    return {"message": "API is responding", "port": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
