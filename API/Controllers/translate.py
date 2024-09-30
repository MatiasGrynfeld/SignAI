import httpx
import os
import asyncio

async def manage_video(url: str, id: str) -> dict:
    base_path = os.getcwd().split("\\")
    project_directory = ""
    for part in base_path:
        if part != "api":
            project_directory += part + "\\"
    
    download_path = project_directory + "AI-Module\\Resources\\Downloads\\" + id + ".mp4"
    
    try:
        if os.path.exists(download_path):
            translation = {"message": "video ya descargado"}
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                with open(download_path, "wb") as f:
                    f.write(response.content)
            translation = {"message": "video descargado correctamente"}

        # Simulación de procesamiento (esto también es asíncrono)
        await asyncio.sleep(5)  # Simulando procesamiento

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:8000/prueba",
                json=translation,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
        
        with open("a.txt", "w") as f:
            f.write(response.text)
        
        return {"message": "Procesamiento y POST completados"}
    
    except Exception as e:
        print(str(e))
        return {"error": str(e)}

async def post_translate(body: dict) -> dict:
    url = body.url
    id = body.id
    
    if not url or not id:
        return {"error": "Faltan datos"}
    
    asyncio.create_task(manage_video(url, id))
    
    return {"message": "llegó", "body": body}
