import pandas as pd

class VideoFormater:
    def __init__(self) -> None:
        pass
    
    def formatVideo(self, points, translation, numVideo):
        dataFrame = pd.DataFrame({'points': points, 'translation': translation})
        dataFrame['id'] = numVideo
        return dataFrame
    
    def concatAndExportVideos(self, videos, file_name):
        concatDf = pd.concat(videos, ignore_index=True)
        concatDf.to_csv(file_name, index=False)

    def csvToDfsVid(self, path):
        df = pd.read_csv(path)
        grouped = df.groupby('id')
        df_list = []
        for _, group in grouped:
            df_list.append(group.reset_index(drop=True))
        return df_list

    def csvToTranslationDf(self, path):
        df = pd.read_csv(path, delimiter='\t')
        df = df.drop(["VIDEO_ID", "SENTENCE_ID", "SENTENCE_NAME", "START", "END"], axis=1)
        df = df.groupby('VIDEO_NAME')['SENTENCE'].apply(' '.join).reset_index()
        return df