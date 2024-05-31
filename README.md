# Cognation X

Cognation X is a sentiment analysis and entity extraction project utilizing a BERT-like model developed with PyTorch. This project is part of the Turkish Natural Language Processing Competition hosted by Teknofest.

## Overview

Cognation X aims to analyze sentiments and extract entities mentioned in Turkish texts. The project leverages a BERT-like model to predict sentiments associated with entities. Additionally, it provides an API using FastAPI to serve predictions.

## Features

- Text data preprocessing
- Training and evaluation of a BERT-like model
- Extraction of entities and their sentiments from text
- Deployment of predictions through a FastAPI-based API
- Swagger UI for API documentation

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/congation-x.git
    cd congation-x
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the necessary spaCy model and NLTK data:
    ```bash
    python -m spacy download tr_core_news_trf
    python -m nltk.downloader punkt
    ```

4. Ensure your trained model is saved as `model.pth` in the project directory.

## Usage

1. Run the FastAPI server:
    ```bash
    uvicorn app:app --reload
    ```

2. Access the Swagger UI to test the API:
    ```
    http://127.0.0.1:8000/docs
    ```

## API Endpoints

### `POST /predict/`

**Request Body**:
```json
{
  "text": "Your input text here"
}
```

**Response**:
```json
{
  "entity_list": ["Entity1", "Entity2", ...],
  "results": [
    {"entity": "Entity1", "sentiment": "olumsuz"},
    {"entity": "Entity2", "sentiment": "nötr"},
    ...
  ]
}
```

## Project Structure

```
.
├── app.py          # FastAPI application
├── ML.py           # Training and evaluation script
├── model.pth       # Trained model file
├── data.csv        # Training data file (if available)
├── README.md       # Project documentation
└── requirements.txt # Python packages dependencies
```

## Training the Model

To train the model, run the `ML.py` script after ensuring you have the necessary data in `data.csv`:

```bash
python ML.py
```

## License

This project is licensed under the MIT License.