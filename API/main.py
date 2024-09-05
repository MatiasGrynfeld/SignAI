from fastapi import FastAPI
from Controllers.root import api_working
from Controllers.translate import post_translate
from Body.translate import BodyTranslate

app = FastAPI()
port = 8000

@app.get("/")
async def root():
    api_working()

@app.post("/translate")
async def translate(body: BodyTranslate):
    data = post_translate(body)
    return {"video_url": body.video_url, "content": data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=port)
