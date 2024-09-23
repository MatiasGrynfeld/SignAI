import httpx
import os

async def post_translate(body):
    url = body.url
    id = body.id
    if not url or not id:
        return {"error": "Faltan datos"}
    base_path = os.getcwd().split("\\")
    project_directory = ""
    for part in base_path:
        if part != "api":
            project_directory += part
            project_directory += "\\"
    download_path = project_directory + "AI-Module\\Resources\\Downloads\\" + id + ".mp4"
    try:
        if os.path.exists(download_path):
            translation = {"message": "Video already downloaded"}
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                with open(download_path, "wb") as f:
                    f.write(response.content)
            
            translation = {"message": "Downloaded video"}
    except Exception as e:
        return {"error": e.__str__()}
    return translation