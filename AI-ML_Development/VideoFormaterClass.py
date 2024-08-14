import pandas as pd

class VideoFormater:
    def __init__(self) -> None:
        pass
    
    def formatVideo(self, points, translation, numVideo):
        dataFrame = pd.DataFrame({'points': points, 'translation': translation})
        dataFrame['id'] = numVideo
        return dataFrame
    
    def concatAndExportVideos(self, videos):
        concatDf = pd.concat(videos, ignore_index=True)
        concatDf.to_csv("videosInfo.csv",index=False)

    def csvToDfs(self, path):
        df = pd.read_csv(path)
        grouped = df.groupby('id')
        df_list = []
        for _, group in grouped:
            df_list.append(group.reset_index(drop=True))
        return df_list
