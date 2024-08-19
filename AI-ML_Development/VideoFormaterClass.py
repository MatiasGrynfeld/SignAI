import pandas as pd

class VideoFormater:
    def __init__(self) -> None:
        pass
    
    def formatVideo(self, dicts):
        df = pd.DataFrame.from_dict(dicts) 
        return df
    
    def concatAndExportVideos(self, df, file_name):
        df.to_csv(file_name, index=False)

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