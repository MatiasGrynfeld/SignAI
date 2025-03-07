from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from API.Controllers.root import api_working
from API.Controllers.translate import post_translate
from API.Body.translate import BodyTranslate
from pydantic import BaseModel

app = FastAPI()
port = 8000

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/")
async def root():
    return api_working()

@app.post("/translate")
async def translate(body: BodyTranslate):
    print("llego algo")
    translation = await post_translate(body)
    return translation

class info(BaseModel):
    id: int
    translation: str

@app.put("/prueba")
async def prueba(body: info):
    print(body)
    return {"received": body}

# Para correr el servidor: uvicorn init:app --host 0.0.0.0 --port 8000 --reload
