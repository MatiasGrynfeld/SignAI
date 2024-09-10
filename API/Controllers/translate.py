import httpx
import os
import sys
base_path = os.getcwd().split("\\")
project_directory = ""
for part in base_path:
    if part != "Controllers":
        project_directory += part
        project_directory += "\\"
sys.path.append(project_directory + "Services")
print(project_directory + "Services")
from Services.clouldinary import get_video_by_id

async def post_translate(body):
    id = body.id
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
            url = get_video_by_id(id)
            #url = "https://campus.ort.edu.ar/static/archivos/videos/mp4/1301449.mp4"
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                with open(download_path, "wb") as f:
                    f.write(response.content)
            
            translation = {"message": "Downloaded video"}
    except:
        return {"error": "Cloudinary not working"}
    return translation