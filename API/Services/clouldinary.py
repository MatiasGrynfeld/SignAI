import os
import cloudinary
cloud_name = os.getenv('CLOUD_NAME')
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')
cloudinary.config(
    cloud_name=cloud_name,
    api_key=api_key,
    api_secret=api_secret,
    secure=True
)
import cloudinary.api as api

def get_video_by_id(id):
    resource = api.resource(id, resource_type="video")
    url = resource["url"]
    return url