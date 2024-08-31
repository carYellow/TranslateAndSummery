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
import asyncio
import logging

# Set environment variables
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA to force CPU usage

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='w')
logger = logging.getLogger(__name__)

# Add a stream handler to also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# Initialize FastAPI app and Jinja2 templates
app = FastAPI()
templates = Jinja2Templates(directory="templatesWithparams")

# Add this after creating the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/staticWithParams", StaticFiles(directory="staticWithParams"), name="staticWithParams")


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

def generate_summary_prompt(text: str) -> str:
    """
    Generate a prompt for summarization.
    
    Args:
    text (str): The text to summarize
    
    Returns:
    str: The generated prompt for the summarization model
    """
    return f"Summarize the following text in exactly 5 bullet points:\n\n{text}\n\nSummary:\n"

async def translate_text_async(text: str, src_lang: str = "heb_Hebr", tgt_lang: str = "eng_Latn") -> str:
    """
    Asynchronous version of translate_text function.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: translator(text, src_lang=src_lang, tgt_lang=tgt_lang)[0]['translation_text'])

async def create_summary_completion_async(
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0
):
    """
    Asynchronous version of create_summary_completion function with tunable parameters.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: phi_3_mini_summarizer(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=["Human:", "Assistant:"],
        echo=False
    ))


@app.get("/summarize")
async def summarize(
    request: Request,
    text: str,
    temperature: float = 0.7,
    max_tokens: int = 400,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0
):
    """
    Endpoint for streaming summary generation and translation to Hebrew with tunable parameters.
    
    Args:
    request (Request): The FastAPI request object
    text (str): The text to summarize
    temperature (float): Controls randomness in generation. Higher values make output more random.
    max_tokens (int): The maximum number of tokens to generate
    top_p (float): Controls diversity via nucleus sampling
    frequency_penalty (float): Penalizes frequent tokens
    presence_penalty (float): Penalizes repeated tokens
    
    Returns:
    StreamingResponse: A streaming response with the generated summary in Hebrew
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")

    try:
        logger.info(f"Received text: {text}")
        logger.info(f"Parameters: temp={temperature}, max_tokens={max_tokens}, top_p={top_p}, "
                    f"freq_penalty={frequency_penalty}, pres_penalty={presence_penalty}")
        
        translated = await translate_text_async(text)
        logger.info(f"Translated text: {translated}")
        summary_prompt = generate_summary_prompt(translated)
        logger.info(f"Generated summary prompt: {summary_prompt}")
        
        summary_result = await create_summary_completion_async(
            summary_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        logger.info(f"Raw summary result: {summary_result}")

        summary = summary_result['choices'][0]['text']
        logger.info(f"Extracted summary: {summary}")
        bullet_points = [point.strip() for point in summary.split('\n') if point.strip()][:5]
        logger.info(f"Extracted bullet points: {bullet_points}")

        async def generate_summary():
            for point in bullet_points:
                hebrew_bullet = await translate_text_async(point, src_lang="eng_Latn", tgt_lang="heb_Hebr")
                logger.info(f"Translated bullet point: {hebrew_bullet}")
                yield f"data: {json.dumps({'he': hebrew_bullet})}\n\n"

        return StreamingResponse(generate_summary(), media_type='text/event-stream')
    except Exception as e:
        logger.error(f"Error in summarize: {str(e)}", exc_info=True)
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
        translated = await translate_text_async(text)
        summary_prompt = generate_summary_prompt(translated)
        summary_result = await create_summary_completion_async(summary_prompt, max_tokens=400)

        summary = summary_result['choices'][0]['text']
        bullet_points = [point.strip() for point in summary.split('\n') if point.strip()][:5]
        formatted_summary = "\n".join(bullet_points)

        return templates.TemplateResponse("result.html", {"request": request, "summary": formatted_summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run("Part 4:app", host="0.0.0.0", port=8000, log_level="debug", reload=True)