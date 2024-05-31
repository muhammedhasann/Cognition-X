import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz.  Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell """)

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    #Buraya model'in çıktısı gelecek
    #Çıktı formatı aşağıdaki örnek gibi olacak
    result = {
  "entity_list": [
    "SuperOnline",
    "Twitch",
    "Kick_Turkey",
    "Başka hiç bir operatörler",
    "Turkcell"
  ],
  "results": [
    {
      "entity": "SuperOnline",
      "sentiment": "olumsuz"
    },
    {
      "entity": "Twitch",
      "sentiment": "nötr"
    },
    {
      "entity": "Kick_Turkey",
      "sentiment": "nötr"
    },
    {
      "entity": " Başka hiç bir operatörler",
      "sentiment": "olumlu"
    },
    {
      "entity": "Turkcell",
      "sentiment": "olumsuz"
    }
  ]
}

    return result


if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)