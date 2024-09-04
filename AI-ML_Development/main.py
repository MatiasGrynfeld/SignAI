from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()
port = 8000

@app.get("/")
async def root():
    return "API Funcionando"

class BodyTranslate(BaseModel):
    video_url: str

@app.post("/translate")
async def translate(body: BodyTranslate):
    url = body.video_url
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()

    return {"video_url": body.video_url, "content": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=port)
