from fastapi import FastAPI
from Controllers.root import api_working
from Controllers.translate import post_translate
from Body.translate import BodyTranslate

app = FastAPI()
port = 8000

@app.get("/")
async def root():
    return api_working()

@app.post("/translate")
async def translate(body: BodyTranslate):
    translation = await post_translate(body)
    return translation

from pydantic import BaseModel
class info(BaseModel):
    info: dict
@app.post("/prueba")
async def prueba(body: info):
    return {"received": body.info}

#uvicorn __init__:app --reload --> to run the server