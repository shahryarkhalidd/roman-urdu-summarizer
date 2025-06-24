from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import T5Tokenizer
from huggingface_hub import hf_hub_download

app = FastAPI()
session = None
tokenizer = None

class InputText(BaseModel):
    text: str

@app.on_event("startup")
def load_model_and_tokenizer():
    global session, tokenizer
    print("🔄 Loading model and tokenizer...")
    
    model_path = hf_hub_download("radientsoul88/roman-urdu-summarizer", "t5_urdu_quant.onnx")
    tokenizer = T5Tokenizer.from_pretrained("radientsoul88/roman-urdu-summarizer")
    
    # Optimize ONNX session with performance settings
    session_options = ort.SessionOptions()
    session_options.inter_op_num_threads = 4  # Adjust based on your CPU cores
    session_options.intra_op_num_threads = 4
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path, 
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )
    
    print("✅ Model and tokenizer loaded. Running warm-up...")
    
    # Warm-up with shorter sequence
    dummy_input = "summarize: test"
    inputs = tokenizer(dummy_input, return_tensors="np", padding="max_length", max_length=32, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    
    # Quick warm-up inference
    try:
        session.run(None, {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
            "decoder_input_ids": decoder_input_ids,
        })
        print("🚀 Warm-up completed successfully.")
    except Exception as e:
        print(f"⚠️ Warm-up failed: {e}")

@app.post("/summarize")
async def summarize(input: InputText):
    global session, tokenizer
    
    # Optimize input processing
    input_text = f"summarize: {input.text}"
    
    # Use shorter max_length for faster processing
    inputs = tokenizer(
        input_text, 
        return_tensors="np", 
        padding="max_length", 
        max_length=64,  # Reduced from 128
        truncation=True
    )
    
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    
    # Reduced max tokens for faster generation
    max_tokens = 10  # Further reduced from 15
    
    try:
        for step in range(max_tokens):  # Fixed syntax error
            outputs = session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                }
            )
            
            logits = outputs[0]
            next_token_logits = logits[:, -1, :]
            
            # Add temperature for faster convergence (optional)
            temperature = 0.7
            next_token_logits = next_token_logits / temperature
            
            # Use argmax for faster deterministic generation
            next_token_id = np.argmax(next_token_logits, axis=-1, keepdims=True)
            
            # Check for end token
            if next_token_id[0, 0] == tokenizer.eos_token_id:
                break
                
            decoder_input_ids = np.hstack([decoder_input_ids, next_token_id])
        
        # Decode summary
        summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
        
        return {"summary": summary.strip()}
        
    except Exception as e:
        return {"error": f"Summarization failed: {str(e)}"}

@app.get("/")
def root():
    return {"message": "Roman Urdu Summarizer API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": session is not None,
        "tokenizer_loaded": tokenizer is not None
    }
