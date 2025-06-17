from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import T5Tokenizer
from scipy.special import softmax
from huggingface_hub import hf_hub_download

app = FastAPI()

# Download model + tokenizer from Hugging Face
model_path = hf_hub_download("radientsoul88/roman-urdu-summarizer", "t5_urdu_quant.onnx")
tokenizer = T5Tokenizer.from_pretrained("radientsoul88/roman-urdu-summarizer")

# Load ONNX session
session = ort.InferenceSession(model_path)

# Input format
class InputText(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(input: InputText):
    input_text = input.text
    inputs = tokenizer("summarize: " + input_text, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    decoder_input_ids = np.array([[tokenizer.pad_token_id]])
    decoder_attention_mask = np.array([[1]])

    max_length = 70
    for _ in range(max_length):
        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask
            }
        )
        logits = outputs[0]
        next_token_logits = logits[:, -1, :]
        probs = softmax(next_token_logits, axis=-1)
        next_token_id = np.array([[np.argmax(probs[0])]])

        decoder_input_ids = np.hstack([decoder_input_ids, next_token_id])
        decoder_attention_mask = np.hstack([decoder_attention_mask, [[1]]])

        if next_token_id[0, 0] == tokenizer.eos_token_id:
            break

    summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    return {"summary": summary}

@app.get("/")
def root():
    return {"message": "Roman Urdu Summarizer API is running"}

