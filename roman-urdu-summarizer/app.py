from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()

# Global cache
model = None
tokenizer = None

class InputText(BaseModel):
    text: str

@app.on_event("startup")
def load_model_and_tokenizer():
    global model, tokenizer
    print("🔄 Loading model and tokenizer...")

    model = T5ForConditionalGeneration.from_pretrained("radientsoul88/urdu")
    tokenizer = T5Tokenizer.from_pretrained("radientsoul88/urdu")

    print("✅ Model and tokenizer loaded.")

@app.post("/summarize")
async def summarize(input: InputText):
    global model, tokenizer

    input_text = input.text
    inputs = tokenizer("summarize: " + input_text, return_tensors="pt", truncation=True)

    summary_ids = model.generate(
        **inputs,
        max_length=60,
        num_beams=2,
        top_k=50,
        top_p=0.95,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

@app.get("/")
def root():
    return {"message": "🚀 Roman Urdu Summarizer API is running"}
