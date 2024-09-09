import httpx
import os

async def post_translate(body):
    id = body.id
    #download_path = "/app/Resources/Downloads/" + id + ".mp4"
    download_path = "C:\\Users\\48519558\\Desktop\\SignAI-ML\\AI-Module\\Resources\\Downloads\\prueba.mp4"
    try:
        # if os.path.exists(download_path):
        #     pass
        # cloud_name = os.getenv('CLOUD_NAME')
        # api_key = os.getenv('API_KEY')
        # api_secret = os.getenv('API_SECRET')
        
        # import cloudinary
        
        # cloudinary.config(
        #     cloud_name=cloud_name,
        #     api_key=api_key,
        #     api_secret=api_secret,
        #     secure=True
        # )
        
        # import cloudinary.api as api
        
        # resource = api.resource(id, resource_type="video")
        # url = resource["url"]
        url = "https://drive.google.com/file/d/1E6_F-Lc8QPXoyw4lHGW7bjvaEvSrsUv3/view?usp=sharing"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            with open(download_path, "wb") as f:
                f.write(response.content)
        
        data = {"message": "Downloaded video"}
    except:
        return {"error": "Cloudinary not working"}
    return data