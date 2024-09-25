import httpx
import os
import asyncio

async def manage_video(url, id):
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

        # Aquí iría la lógica para procesar el video
        await asyncio.sleep(5)  # Simulando un procesamiento que toma tiempo
        async with httpx.AsyncClient() as client:
            response = client.post(
                "url",
                json=translation,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            f = open("a.txt", "w")
            f.write(response)
            f.close()
            return
    except Exception as e:
        return {"error": str(e)}

async def post_translate(body):
    url = body.url
    id = body.id
    if not url or not id:
        return {"error": "Faltan datos"}
    
    asyncio.create_task(manage_video(url, id))
    return {"message": "procesado correct", "body": body}