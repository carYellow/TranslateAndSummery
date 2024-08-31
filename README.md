
This FastAPI application provides an AI-powered text summarization and translation service. It uses the Phi-3 mini model for summarization and the NLLB-200 model for translation.

## Features

- Summarize text into 5 bullet points
- Translate text between Hebrew and English
- Stream summarized and translated results
- Web interface for easy interaction

## Requirements

- Python 3.7+
- FastAPI
- Transformers
- llama-cpp-python
- Jinja2

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the Phi-3 mini model

## Usage

1. Start the server: `python Part_4.py`
2. Access the web interface at `http://localhost:8000`
3. Use the `/summarize` endpoint for API access

## API Endpoints

- GET `/`: Main page
- GET `/summarize`: Summarize and translate text (streaming)
- POST `/summarize_form`: Handle form submissions

## Configuration

- Adjust model parameters in the `summarize` function
- Modify logging settings as needed

## License

[Your chosen license]
