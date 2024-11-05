from pydantic import BaseModel

class BodyTranslate(BaseModel):
    id: int
    url: str
    
"""
BodyTranslate Example:

{
    "id": "1",
    "url": "https://www.youtube.com/watch?v=9bZkp7q19f0"
}
"""