import json
import pickle
from annoy import AnnoyIndex
import mlflow.sklearn
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
import faiss
from scipy.sparse import csr_matrix
import numpy as np

mlflow.set_tracking_uri("../../deployment/mlruns/")
mlflow.set_experiment(experiment_name = "recommendation_model")

def create_matrix(df):
    """
    Creates a sparse matrix from a dataframe

    Args:
        df: pandas dataframe

    Returns:
        matrix: sparse matrix
    """
    U = df["userId"].nunique()
    M = df["imdbId"].nunique()

    userId_mapper = dict(zip(np.unique(df["userId"]), list(range(U))))
    imdbId_mapper = dict(zip(np.unique(df["imdbId"]), list(range(M))))

    user_index = [userId_mapper[i] for i in df["userId"]]
    imdbId_index = [imdbId_mapper[i] for i in df["imdbId"]]

    matrix = csr_matrix((df["rating"], (user_index, imdbId_index)), shape=(U,M))

    return matrix

def training(matrix, features_df):
    """
    Trains an ANNOY index and a FAISS index from 2 datasets different datasets

    Args:
        matrix: Sparse matrix for collaborative filtering 
        features_df: Dataframes with the genres and decades from each movie

    Returns:
        faiss_index: FAIIS index trained
        annoy_index: ANNOY index trained
        features_dense: Features_df where we applied PCA (Dimensionality Reduction)
        dim: Dimension used in Annoy Index
        annoy_metric: Metric used in Annoy Index
    """
    #FAISS
    features = features_df.values.astype("float32")

    pca_full = PCA()
    pca_full.fit_transform(features)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    threshold = 0.95
    n_components = np.argmax(cumulative_variance >= threshold) + 1

    pca = PCA(n_components=n_components)
    features_dense = pca.fit_transform(features)

    dimension = features_dense.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(features_dense)

    #ANNOY
    annoy_metric = "angular"

    svd_full = TruncatedSVD()
    svd_full.fit_transform(matrix.T)
    cumulative_variance = np.cumsum(svd_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= threshold) + 1

    svd = TruncatedSVD(n_components=n_components)
    embeddings = svd.fit_transform(matrix.T)
    dim = n_components
    annoy_index = AnnoyIndex(dim,annoy_metric)
    for idx, embedding in enumerate(embeddings):
        annoy_index.add_item(idx,embedding)
    annoy_index.build(20)

    return faiss_index, annoy_index, features_dense, dim, annoy_metric

def savings(faiss_index, annoy_index, features_dense, movie_idx, movies, dim, annoy_metric):
    """
    Saves every index and parameter necessary for the api/app in their respective folder
    """
    faiss.write_index(faiss_index, "../model/faiss_index.bin")
    annoy_index.save("../model/annoy_index.ann")

    with open("../dictionaries/features_dense.pkl", "wb") as handle:
        pickle.dump(features_dense, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../dictionaries/movie_idx.pkl", "wb") as handle:
        pickle.dump(movie_idx, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../dictionaries/movies.pkl", "wb") as handle:
        pickle.dump(movies, handle, protocol = pickle.HIGHEST_PROTOCOL)

    annoy_config = {
        "dimension":dim,
        "metric": annoy_metric,
    }
    with open("../model/annoy_config.json","w") as handle:
        json.dump(annoy_config,handle,indent=4)

    mlflow.sklearn.log_model(faiss_index,"faiss_model")


if __name__ == "__main__":
    data = pd.read_csv("../datasets/processed_data/processedData.csv", sep = ",")
    features_df = pd.read_csv("../datasets/processed_data/featuresDataset.csv", sep = ",")

    movies = features_df[["title","imdbId"]]
    features_df.drop(columns=["title","imdbId"], axis = 1, inplace = True)
    movies = pd.DataFrame(movies)
    movie_idx = dict(zip(movies["title"],list(movies.index)))

    matrix = create_matrix(data)
    faiss_index, annoy_index, features_dense, dim, annoy_metric = training(matrix, features_df)
    savings(faiss_index,annoy_index,features_dense,movie_idx,movies,dim,annoy_metric)
