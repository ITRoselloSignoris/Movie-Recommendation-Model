import pandas as pd
import pickle
import os
import uvicorn
import faiss
from annoy import AnnoyIndex
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import json

PARENT_FOLDER = os.path.dirname(__file__)

FAISS_PATH = os.path.join(PARENT_FOLDER, "../src/model/faiss_index.bin")
ANNOY_PATH = os.path.join(PARENT_FOLDER, "../src/model/annoy_index.ann")

ANNOY_CONFIG_PATH = os.path.join(PARENT_FOLDER, "../src/model/annoy_config.json")

IDX_PATH = os.path.join(PARENT_FOLDER, "../src/dictionaries/movie_idx.pkl")
MOVIES_PATH = os.path.join(PARENT_FOLDER, "../src/dictionaries/movies.pkl")
FEATURES_PATH = os.path.join(PARENT_FOLDER, "../src/dictionaries/features_dense.pkl")

with open(ANNOY_CONFIG_PATH,"r") as handle:
    annoy_config = json.load(handle)

with open(IDX_PATH, "rb") as handle:
    movie_idx = pickle.load(handle)

with open(MOVIES_PATH, "rb") as handle:
    movies = pickle.load(handle)

with open(FEATURES_PATH, "rb") as handle:
    embeddings = pickle.load(handle)

ID_USER = os.getenv("ID_USER", "Desconocida/o")

app = FastAPI()

class Input_data(BaseModel):
    model: str
    n_recommendations: int
    movie_title: str

@app.get("/")
async def root():
    return {"message": "Usuario/a " + ID_USER}

@app.post("/recommendation")
def get_recommendation(input_data:Input_data):
    similar_indices=[]
    similar_movies=[]
    response = "Movie Unrecognized. Try Again"
    answer_dict = jsonable_encoder(input_data)
    for key, value in answer_dict.items():
        answer_dict[key] = value
    model = answer_dict["model"].upper()
    n_recommendations = answer_dict["n_recommendations"]
    title = answer_dict["movie_title"]
    if title not in movie_idx.keys():
        return response
    idx = movie_idx[title]
    if model == "FAISS":
        faiss_index = faiss.read_index(FAISS_PATH)
        query = embeddings[idx].reshape(1,-1)
        distances, indices = faiss_index.search(query,k=n_recommendations+1)
        similar_indices = indices[0].tolist()
    elif model == "ANNOY":
        dimension = annoy_config["dimension"]
        metric = annoy_config["metric"]
        annoy_index = AnnoyIndex(dimension,metric)
        annoy_index.load(ANNOY_PATH)
        similar_indices = annoy_index.get_nns_by_item(idx, n_recommendations+1)
    if idx in similar_indices:
        similar_indices.remove(idx)
    similar_indices=similar_indices[:n_recommendations]
    similar_movies = movies["title"].iloc[similar_indices]
    response = f"({model} model)\nIf you liked {title}, you should watch:\n{similar_movies}"
    return response

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 7860)