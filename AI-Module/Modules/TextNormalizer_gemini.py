import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()
class TextNormalizer:
    def __init__(self, api_key):
        # Configurar la API Key
        genai.configure(api_key=api_key)
        generation_config={
            "temperature":1,
            "top_p":0.95,
            "top_k":64,
            "max_output_tokens":8192,
            "response_mime_type":"text/plain"
        }
        self.model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)
    def normalizar_texto(self, output):
            
            chat_session= self.model.start_chat(
                history=[
                    {
                    "role":"user",
                    "parts":[
                        "You are going to recieve a text that may not be gramatically correct, and it also may be robotic, normalize it so it has the same meaning"
                    ]
                    }, 
                ]
            )
            response=chat_session.send_message(output)

            content=response.candidates[0].content
            parts=content.parts
            response=parts[0].text
            return response

api_key=os.getenv("API_KEY")
print(api_key)
normalizer=TextNormalizer(api_key)
result=normalizer.normalizar_texto("I'm football player good")
print(result)
