import os
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from llama_cpp import Llama
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Set environment variables
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA to force CPU usage

# Initialize FastAPI app and Jinja2 templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Add this after creating the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Load translation model
print("Loading translation model...")
translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
# Create a translation pipeline, forcing CPU usage
translator = pipeline("translation", model=translator_model, tokenizer=translator_tokenizer, device=-1)

# Load Llama model for summarization
print("Loading Phi-3 mini model...")
phi_3_mini_summarizer = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=-1,  # Use all available layers on GPU
    n_ctx=2048,  # Set context size to 2048 tokens
    n_batch=512,  # Set batch size to 512 for better performance
    use_mlock=True,  # Pin the model in memory to prevent swapping
    use_mmap=True,  # Use memory mapping for faster loading
)

# Define request model for summarization
class SummarizeRequest(BaseModel):
    text: str

def translate_text(text: str, src_lang: str = "heb_Hebr", tgt_lang: str = "eng_Latn") -> str:
    """
    Translate text from source language to target language.
    
    Args:
    text (str): The text to translate
    src_lang (str): The source language code (default: Hebrew)
    tgt_lang (str): The target language code (default: English)
    
    Returns:
    str: The translated text
    """
    return translator(text, src_lang=src_lang, tgt_lang=tgt_lang)[0]['translation_text']

def generate_summary_prompt(text: str) -> str:
    """
    Generate a prompt for summarization.
    
    Args:
    text (str): The text to summarize
    
    Returns:
    str: The generated prompt for the summarization model
    """
    return f"Summarize the following text in exactly 5 bullet points:\n\n{text}\n\nSummary:\n"

def create_summary_completion(prompt: str, max_tokens: int = 300, temperature: float = 0.7, stream: bool = True):
    """
    Create a chat completion for summarization using the Phi-3 mini model.
    
    Args:
    prompt (str): The summarization prompt
    max_tokens (int): Maximum number of tokens to generate
    temperature (float): Sampling temperature (higher = more random)
    stream (bool): Whether to stream the response or not
    
    Returns:
    dict or generator: The model's response
    """
    return phi_3_mini_summarizer.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text in exactly 5 bullet points."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream
    )

@app.get("/summarize")
async def summarize(request: Request):
    """
    Endpoint for streaming summary generation.
    
    Args:
    request (Request): The FastAPI request object
    
    Returns:
    StreamingResponse: A streaming response with the generated summary
    """
    text = request.query_params.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")

    try:
        translated = translate_text(text)
        summary_prompt = generate_summary_prompt(translated)

        def generate_summary():
            summary_result = create_summary_completion(summary_prompt)
            current_bullet = ""
            bullet_count = 0
            for chunk in summary_result:
                if 'content' in chunk['choices'][0]['delta']:
                    content = chunk['choices'][0]['delta']['content']
                    current_bullet += content
                    if '\n' in content or '•' in content or len(current_bullet) > 100:
                        yield f"data: {json.dumps({'content': current_bullet.strip()})}\n\n"
                        current_bullet = ""
                        bullet_count += 1
                        if bullet_count >= 5:
                            break
            if current_bullet:
                yield f"data: {json.dumps({'content': current_bullet.strip()})}\n\n"

        return StreamingResponse(generate_summary(), media_type='text/event-stream')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Render the main page.
    
    Args:
    request (Request): The FastAPI request object
    
    Returns:
    TemplateResponse: The rendered HTML template for the main page
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize_form", response_class=HTMLResponse)
async def summarize_form(request: Request, text: str = Form(...)):
    """
    Handle form submission for summarization.
    
    Args:
    request (Request): The FastAPI request object
    text (str): The text to summarize, received from the form
    
    Returns:
    TemplateResponse: The rendered HTML template with the summary result
    """
    try:
        translated = translate_text(text)
        summary_prompt = generate_summary_prompt(translated)
        summary_result = create_summary_completion(summary_prompt, max_tokens=400, stream=False)

        summary = summary_result['choices'][0]['message']['content']
        bullet_points = []
        current_bullet = ""
        for line in summary.split('\n'):
            if line.strip().startswith('•') or line.strip().startswith('-'):
                if current_bullet:
                    bullet_points.append(current_bullet.strip())
                current_bullet = line.strip()
            else:
                current_bullet += ' ' + line.strip()
        if current_bullet:
            bullet_points.append(current_bullet.strip())

        formatted_summary = "\n".join(bullet_points[:5])  # Ensure we only have 5 bullet points

        return templates.TemplateResponse("result.html", {"request": request, "summary": formatted_summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)