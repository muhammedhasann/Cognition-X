import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import spacy

# Load spaCy model
nlp = spacy.load('tr_core_news_trf')

# Load the trained model and tokenizer
model = torch.load('model.pth')
model.eval()
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

class TextRequest(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz. Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell""")

app = FastAPI()

@app.post("/predict/")
async def predict(request: TextRequest):
    text = request.text
    
    # Extract entities
    entities = [ent.text for ent in nlp(text).ents]
    
    # Tokenize and predict sentiment for each entity
    sentiments = []
    for entity in entities:
        inputs = tokenizer(entity, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        sentiment = torch.argmax(logits, dim=1).item()
        sentiments.append(sentiment)

    # Map sentiment to string labels
    sentiment_labels = ["olumsuz", "nötr", "olumlu"]
    results = [{"entity": entity, "sentiment": sentiment_labels[sentiment]} for entity, sentiment in zip(entities, sentiments)]

    return {"entity_list": entities, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
