from Modules.KeyFrameExtractorClass import KeyFrameExtractor
from Modules.Points2VecClass import Point2Vec
from Modules.TextNormalizer_gemini import TextNormalizer
from Modules.cnn_lstmCLass import Model
import cv2
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def main(path:str) -> str:
    path = Path(path)
    kfe = KeyFrameExtractor()
    normalizer = Point2Vec(4)
    api_key=os.getenv("API_KEY")
    text_normalizer=TextNormalizer(api_key)
    tokenizer_path='./tokenizer.json'
    model_path='./model_seq2seq_with_attention_and_word2vec_2.h5'
    model= Model(model_path,tokenizer_path)
    
    try:
        #Use ffmpeg to filter video frames
        
        path = ffmpeg_video(path)
        # return "Hello World"
        
        #Create video object
        
        video = cv2.VideoCapture(str(path))
        
        #Extract video features
        
        video_features = kfe.extractKeyFrames(return_frame=False, draw=False, video=video)
        
        #Normalize video features
        
        normalized_video_features = normalizer.land2vec(video_features)

        #Modelo
        output= model.predict(normalized_video_features)

        #Normalize text
        text=text_normalizer.normalizar_texto(output)
        return text
        
    except:
        raise Exception("An error occurred")

def ffmpeg_video(path: str) -> Path:
    if path.suffix == ".mp4":
        folder = path.parent
        file_name = path.name
        output_path = folder / f"filtered_{file_name}"
        
        # Comando ffmpeg
        command = [
            "ffmpeg", 
            "-i", str(path), 
            "-vf", r"select=not(mod(n\,3))", 
            "-vsync", "vfr", 
            "-c:v", "libx264", 
            str(output_path)
        ]
        subprocess.run(command, check=True)
        return output_path
    else:
        raise Exception("Invalid file format")

if __name__ == "__main__":
    path = input("Enter the path of the video: ")
    print(main(path))