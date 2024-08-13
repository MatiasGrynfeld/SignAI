import pandas as pd

class VideoFormater:
    def __init__(self) -> None:
        pass
    
    def formatVideo(self, points, numVideo):
        dataFrame = pd.DataFrame({'puntos': points})
        dataFrame['id'] = numVideo
        return dataFrame
    
    def concatAndExportVideos(self, videos):
        concatDf = pd.concat(videos, ignore_index=True)
        concatDf.to_csv("videosInfo.csv",index=False)
