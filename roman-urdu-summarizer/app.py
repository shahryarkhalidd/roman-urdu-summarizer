from fastapi import FastAPI
from pydantic import BaseModel
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Global variables for models
model_engine = None
executor = ThreadPoolExecutor(max_workers=1)

class InputText(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    global model_engine
    print("🔄 Installing and loading optimized T5 model...")
    
    try:
        # Import fastT5 - this handles the ONNX optimization properly
        from fastt5 import OnnxT5
        
        print("🔄 Loading model with fastT5 (optimized ONNX with KV-cache)...")
        
        # Load the model with proper ONNX optimization
        model_engine = OnnxT5.from_pretrained("radientsoul88/roman-urdu-summarizer")
        
        print("✅ Model loaded successfully with fastT5 optimization")
        
        # Warm-up
        print("🔥 Running warm-up...")
        test_summary = model_engine("summarize: this is a test message", max_length=20)
        print(f"🚀 Warm-up completed. Test output: {test_summary}")
        
    except ImportError:
        print("❌ fastT5 not installed. Installing now...")
        import subprocess
        import sys
        
        # Install fastT5
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastt5"])
        
        # Try again
        from fastt5 import OnnxT5
        model_engine = OnnxT5.from_pretrained("radientsoul88/roman-urdu-summarizer")
        print("✅ fastT5 installed and model loaded")
        
    except Exception as e:
        print(f"⚠️ fastT5 failed, trying alternative approach: {e}")
        
        # Fallback to onnxt5 library
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxt5"])
            
            from onnxt5 import GenerativeT5
            from onnxt5.api import get_encoder_decoder_tokenizer
            
            # Load with onnxt5 (alternative optimized library)
            model_path = "radientsoul88/roman-urdu-summarizer"
            encoder, decoder, tokenizer = get_encoder_decoder_tokenizer(model_path)
            model_engine = GenerativeT5(encoder, decoder, tokenizer)
            print("✅ Model loaded with onnxt5 optimization")
            
        except Exception as e2:
            print(f"❌ All optimized approaches failed: {e2}")
            print("🔄 Loading PyTorch model as final fallback...")
            
            # Final fallback to PyTorch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import torch
            
            global tokenizer
            tokenizer = T5Tokenizer.from_pretrained("radientsoul88/roman-urdu-summarizer")
            model_engine = T5ForConditionalGeneration.from_pretrained("radientsoul88/roman-urdu-summarizer")
            model_engine.eval()
            
            # Enable optimizations
            if torch.cuda.is_available():
                model_engine = model_engine.to("cuda")
                print("✅ PyTorch model loaded on GPU")
            else:
                model_engine = model_engine.to("cpu")
                print("✅ PyTorch model loaded on CPU")

def generate_summary_sync(input_text: str) -> str:
    """Synchronous function to run in thread pool"""
    try:
        # Check which model type we're using
        if hasattr(model_engine, '__call__') and not hasattr(model_engine, 'generate'):
            # fastT5 or onnxt5
            summary = model_engine(f"summarize: {input_text}", max_length=30, min_length=5)
            return summary if isinstance(summary, str) else summary[0]
            
        else:
            # PyTorch model
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained("radientsoul88/roman-urdu-summarizer")
            
            inputs = tokenizer(
                f"summarize: {input_text}",
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            )
            
            # Move to same device as model
            device = next(model_engine.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model_engine.generate(
                    **inputs,
                    max_length=30,
                    min_length=5,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False,
                    length_penalty=1.0
                )
            
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()
            
    except Exception as e:
        return f"Generation failed: {str(e)}"

@app.post("/summarize")
async def summarize(input: InputText):
    if model_engine is None:
        return {"error": "Model not loaded"}
    
    start_time = time.time()
    
    try:
        # Run the generation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(executor, generate_summary_sync, input.text)
        
        generation_time = time.time() - start_time
        
        return {
            "summary": summary,
            "generation_time_seconds": round(generation_time, 2),
            "model_type": type(model_engine).__name__
        }
        
    except Exception as e:
        return {
            "error": f"Summarization failed: {str(e)}",
            "generation_time_seconds": round(time.time() - start_time, 2)
        }

@app.get("/")
def root():
    return {"message": "Optimized Roman Urdu Summarizer API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_engine is not None,
        "model_type": type(model_engine).__name__ if model_engine else None
    }

# Alternative endpoint for batch processing
@app.post("/batch_summarize")
async def batch_summarize(inputs: list[InputText]):
    if model_engine is None:
        return {"error": "Model not loaded"}
    
    start_time = time.time()
    results = []
    
    for item in inputs:
        try:
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(executor, generate_summary_sync, item.text)
            results.append({"text": item.text, "summary": summary, "success": True})
        except Exception as e:
            results.append({"text": item.text, "error": str(e), "success": False})
    
    total_time = time.time() - start_time
    
    return {
        "results": results,
        "total_time_seconds": round(total_time, 2),
        "average_time_per_item": round(total_time / len(inputs), 2)
    }
