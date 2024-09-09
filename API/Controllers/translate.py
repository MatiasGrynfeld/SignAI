import httpx
import os

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
            cloud_name = os.getenv('CLOUD_NAME')
            api_key = os.getenv('API_KEY')
            api_secret = os.getenv('API_SECRET')
            
            import cloudinary
            
            cloudinary.config(
                cloud_name=cloud_name,
                api_key=api_key,
                api_secret=api_secret,
                secure=True
            )
            
            import cloudinary.api as api
            
            resource = api.resource(id, resource_type="video")
            url = resource["url"]
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                with open(download_path, "wb") as f:
                    f.write(response.content)
            
            translation = {"message": "Downloaded video"}
    except:
        return {"error": "Cloudinary not working"}
    return translation