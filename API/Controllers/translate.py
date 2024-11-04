import httpx
import os
import asyncio
from pathlib import Path
import sys

project_directory = Path(os.getcwd()).parent / "AI-Module"
sys.path.append(str(project_directory))
sys.path.append(str(project_directory / "Modules"))
print(sys.path)

import main as translate

async def manage_video(id:int, path:str):
    if id=="video_prueba":
        id=-100
    try:
        print("entre", id, path)
        print(type(translate))
        translation = translate(path)
        print("pre")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://sign-ai-web.vercel.app/{id}/texto", #Cambiar a la ruta del back
                json={"id":id,"translation": translation},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
        print("mande")
        return
    except Exception as e:
        print(str(e))
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://sign-ai-web.vercel.app/{id}/texto", #Cambiar a la ruta del back
                json={"id":id,"translation": "Error"},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
        return {"error": str(e)}

async def post_translate(body: dict) -> dict:
    url = body.url
    id = body.id
    
    if not url or not id:
        return {"error": "Faltan datos"}
    download = project_directory / "Resources" / "Downloads"
    download_path = project_directory / "Resources" / "Downloads" / f"{id}.mp4"
    try:
        already_downloaded = False
        if not os.path.exists(download_path):
            os.makedirs(str(download))
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                with open(download_path, "wb") as f:
                    f.write(response.content)
        else:
            already_downloaded = True
        asyncio.create_task(manage_video(id, download_path))
        return {"message": "received", "body": body, "already_downloaded": already_downloaded}
    except Exception as e:
        print(str(e))
        return {"error": str(e)}
