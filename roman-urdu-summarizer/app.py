from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import T5Tokenizer
from huggingface_hub import hf_hub_download

app = FastAPI()

# Global cache variables
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
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print("✅ Model and tokenizer loaded.")

@app.post("/summarize")
async def summarize(input: InputText):
    global session, tokenizer

    # Prepare input
    input_text = "summarize: " + input.text
    inputs = tokenizer(input_text, return_tensors="np", padding="max_length", max_length=512, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    decoder_input_ids = np.array([[tokenizer.pad_token_id]])
    max_length = 30

    for _ in range(max_length):
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
        next_token_id = np.argmax(next_token_logits, axis=-1, keepdims=True)

        decoder_input_ids = np.hstack([decoder_input_ids, next_token_id])

        if next_token_id[0, 0] == tokenizer.eos_token_id:
            break

    summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    return {"summary": summary}

@app.get("/")
def root():
    return {"message": "Roman Urdu Summarizer API is running"}
