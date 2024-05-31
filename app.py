import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from model import BERT, preprocess_text, text_to_sequence, predict_entities_and_sentiments, word2idx, device

app = FastAPI()

# Example input data model
class Item(BaseModel):
    text: str = Field(
        ...,
        example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz. Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim? @Turkcell"""
    )

# Initialize model
vocab_size = len(word2idx)
embed_dim = 128
num_heads = 8
feedforward_dim = 512
num_layers = 6
max_len = 512
model = BERT(vocab_size, embed_dim, num_heads, feedforward_dim, num_layers, max_len).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))  # Load your trained model
model.eval()

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    # Predict entities and sentiments
    result = predict_entities_and_sentiments(item.text, model, word2idx, device)
    
    # Return the result in the required format
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
