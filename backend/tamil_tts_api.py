from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
try:
    from transformers import VitsModel, AutoTokenizer
    import torch
    import scipy.io.wavfile
    import io
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Tamil TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    if HAS_DEPS:
        print("Loading facebook/mms-tts-tam model...")
        model = VitsModel.from_pretrained("facebook/mms-tts-tam")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tam")
        print("Model loaded successfully.")
    else:
        print("Transformers or SciPy dependencies missing.")

@app.get("/health")
def health_check():
    return {"status": "ok", "loaded": model is not None}

@app.get("/tts")
def generate_tts(text: str):
    if not model or not tokenizer:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    
    if not text.strip():
        return JSONResponse({"error": "No text provided"}, status_code=400)

    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Convert to WAV in memory
        wav_io = io.BytesIO()
        scipy.io.wavfile.write(wav_io, rate=model.config.sampling_rate, data=output[0].cpu().numpy())
        wav_io.seek(0)
        
        return StreamingResponse(wav_io, media_type="audio/wav")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Make sure we import dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    uvicorn.run("tamil_tts_api:app", host="0.0.0.0", port=9010, reload=True)
