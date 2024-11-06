from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Controllers.root import api_working
from Controllers.translate import post_translate
from Body.translate import BodyTranslate
from pydantic import BaseModel

app = FastAPI()
port = 8000

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las solicitudes de cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
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

@app.post("/prueba")
async def prueba(body: info):
    print(body)
    return {"received": body}

# Para correr el servidor: uvicorn init:app --host 0.0.0.0 --port 8000 --reload
