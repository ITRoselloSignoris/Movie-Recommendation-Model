import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def agregate_genres(genres):
    union=set()
    for genre_list in genres:
        union.update(genre_list)
    return list(union)

def data_preparation(data, genres_df):
    """
    Applies data preparation techniques in both datasets. 
    In the end, it saves both modified datasets.

    Args:
        data: Dataset with ratings 
        genres_df: Genres Dataframe
    """ 
    decades=[]
    for x in genres_df["year"]:
        if str(x) == "nan":
            decades.append("unknown")
        else:
            decades.append(str(x)[0:3]+"0s")

    genres_df["decade"] = decades
    genres_df = genres_df.drop("year",axis=1)

    genres_df["imdbId"] = genres_df["imdbId"].astype("str")
    df = genres_df[["imdbId","title","decade"]].drop_duplicates(subset=["title"])

    genres_df = genres_df.groupby("title",as_index=False).agg({"genres":agregate_genres})
    genres_df=pd.merge(genres_df,df, on="title",how="left")
    genres_df = genres_df.drop_duplicates(subset=["title"])

    #One Hot Encoding in genres column and decades column
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(genres_df["genres"])
    df_genres = pd.DataFrame(matrix, columns=mlb.classes_)
    genres_df = pd.concat([genres_df[["imdbId","title","decade"]],df_genres],axis=1)
    decades_category = pd.get_dummies(genres_df["decade"])

    features_df = genres_df.drop(columns = "decade", axis = 1)
    features_df = pd.concat([features_df,decades_category],axis=1)

    #Saving Datasets
    features_df.to_csv("../datasets/processed_data/featuresDataset.csv", index = False)
    data.to_csv("../datasets/processed_data/processedData.csv", index = False)

if __name__ == "__main__":
    data = pd.read_csv("../datasets/adapted_data/adaptedRatingsData.csv", sep = ",")
    genres_df = pd.read_csv("../datasets/adapted_data/adaptedFeaturesData.csv", sep = ",")
    genres_df['genres'] = genres_df['genres'].apply(ast.literal_eval)
    data_preparation(data, genres_df)