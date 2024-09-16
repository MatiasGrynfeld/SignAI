import cloudinary
import os
from dotenv import load_dotenv

load_dotenv()

cloud_name = os.getenv('CLOUD_NAME')
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

# Configuration       
cloudinary.config( 
    cloud_name = cloud_name, 
    api_key = api_key, 
    api_secret = api_secret,
    secure=True
)

import cloudinary.uploader

# Upload an image
upload_result = cloudinary.uploader.upload(
    "C:\\Users\\48519558\\Desktop\\SignAI-ML\\AI-Module\\Resources\\Videos\\_2FBDaOPYig-5-rgb_front.mp4", 
    public_id="video_prueba",
    resource_type = "video"
)
print(upload_result["secure_url"])