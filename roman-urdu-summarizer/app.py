from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from huggingface_hub import hf_hub_download

app = FastAPI()

# Global cache variables
model = None
tokenizer = None

class InputText(BaseModel):
    text: str

@app.on_event("startup")
def load_model_and_tokenizer():
    global model, tokenizer
    print("🔄 Loading optimized model and tokenizer...")

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("radientsoul88/roman-urdu-summarizer")

    # Load ORT model
    model = ORTModelForSeq2SeqLM.from_pretrained("radientsoul88/roman-urdu-summarizer", export=False)
    print("✅ Optimized model and tokenizer loaded.")

@app.post("/summarize")
async def summarize(input: InputText):
    global model, tokenizer

    input_text = input.text
    inputs = tokenizer("summarize: " + input_text, return_tensors="pt")

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=40,
        num_beams=2,
        early_stopping=True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"summary": summary}

@app.get("/")
def root():
    return {"message": "Roman Urdu Summarizer API is running"}
