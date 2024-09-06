import httpx
import time

async def post_translate(body):
    url = body.video_url
    async with httpx.AsyncClient() as client:
        video = await client.get(url)
        data = video.json()
    return data