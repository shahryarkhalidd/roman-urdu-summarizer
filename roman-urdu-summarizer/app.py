from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import T5Tokenizer
from scipy.special import softmax
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
    session = ort.InferenceSession(model_path)

    print("✅ Model and tokenizer loaded.")

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95):
    """Apply top-k and top-p (nucleus) filtering to logits."""
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    cumulative_probs = np.cumsum(softmax(sorted_logits))

    # Keep tokens with cumulative probs < top_p
    cutoff_index = np.searchsorted(cumulative_probs, top_p) + 1
    top_indices = sorted_indices[:max(top_k, cutoff_index)]

    filtered_probs = softmax(logits[top_indices])
    filtered_probs /= filtered_probs.sum()  # Renormalize

    return top_indices, filtered_probs

@app.post("/summarize")
async def summarize(input: InputText):
    global session, tokenizer

    input_text = input.text
    inputs = tokenizer("summarize: " + input_text, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    beam_width = 2
    max_length = 60
    top_k = 50
    top_p = 0.95

    # Initialize beams
    beams = [{
        "decoder_input_ids": np.array([[tokenizer.pad_token_id]]),
        "decoder_attention_mask": np.array([[1]]),
        "score": 0.0
    }]

    for _ in range(max_length):
        all_candidates = []

        for beam in beams:
            decoder_input_ids = beam["decoder_input_ids"]
            decoder_attention_mask = beam["decoder_attention_mask"]

            outputs = session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask
                }
            )

            logits = outputs[0][:, -1, :][0]  # (vocab_size,)
            top_indices, filtered_probs = top_k_top_p_filtering(logits, top_k, top_p)

            for idx, prob in zip(top_indices, filtered_probs):
                new_decoder_input_ids = np.hstack([decoder_input_ids, [[idx]]])
                new_decoder_attention_mask = np.hstack([decoder_attention_mask, [[1]]])

                new_score = beam["score"] + np.log(prob + 1e-8)  # accumulate log probs
                all_candidates.append({
                    "decoder_input_ids": new_decoder_input_ids,
                    "decoder_attention_mask": new_decoder_attention_mask,
                    "score": new_score
                })

        # Keep top beams
        beams = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:beam_width]

        # Early stopping if all beams ended
        if all(beam["decoder_input_ids"][0, -1] == tokenizer.eos_token_id for beam in beams):
            break

    # Pick best beam
    best_beam = beams[0]
    summary = tokenizer.decode(best_beam["decoder_input_ids"][0], skip_special_tokens=True)
    return {"summary": summary}

@app.get("/")
def root():
    return {"message": "Roman Urdu Summarizer API is running"}
