import pandas as pd
import pickle
import os
import faiss
from annoy import AnnoyIndex
import json
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from fuzzywuzzy import process, fuzz

PARAMS_NAME = [
    "index",
    "n_recommendations",
    "movie_title"
    ]

PARENT_FOLDER = os.path.dirname(__file__)

FAISS_PATH = os.path.join(PARENT_FOLDER, "index/faiss_index.bin")
ANNOY_PATH = os.path.join(PARENT_FOLDER, "index/annoy_index.ann")

ANNOY_CONFIG_PATH = os.path.join(PARENT_FOLDER, "index/annoy_config.json")

IDX_PATH = os.path.join(PARENT_FOLDER, "dictionaries/movie_idx.pkl")
MOVIES_PATH = os.path.join(PARENT_FOLDER, "dictionaries/movies.pkl")
FEATURES_PATH = os.path.join(PARENT_FOLDER, "dictionaries/features_dense.pkl")

with open(ANNOY_CONFIG_PATH,"r") as handle:
    annoy_config = json.load(handle)

with open(IDX_PATH, "rb") as handle:
    movie_idx = pickle.load(handle)

with open(MOVIES_PATH, "rb") as handle:
    movies = pickle.load(handle)

with open(FEATURES_PATH, "rb") as handle:
    embeddings = pickle.load(handle)

def movie_finder(title):
    """
    Finds the closest match to a given string.

    Args:
        title: Movie title string
    """
    threshold = 70
    all_titles = movies["title"].tolist()
    closest_match = process.extractOne(title,all_titles,scorer=fuzz.WRatio, score_cutoff=threshold)
    return closest_match[0] if closest_match else None

def get_plot(state):
    query_idx=state["movie_idx"]
    rec_indices=state["similar_indices"]
    movie_title = state["movie_title"]
    index = state["index"]

    pca=PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    fig, ax=plt.subplots(figsize=(10,8))
    ax.scatter(embeddings_2d[:,0],embeddings_2d[:,1],color="lightgray",label="Movies",alpha=0.6, s=40)
    ax.scatter(embeddings_2d[query_idx,0],embeddings_2d[query_idx,1], marker='*', s=200,color="red",label=f"Query: {movie_title}")
    ax.scatter(embeddings_2d[rec_indices,0],embeddings_2d[rec_indices,1],color="blue",s=80,label=f"{index} neighbors")
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title(f"{index} Recommendations for {movie_title}")
    ax.legend()

    return fig

def make_dict(ids,titles):#
    """"
    Creates a dictionary from ids list and titles list.

    Args:
        ids: List with Imdb ids
        titles: List with Movie Titles

    Returns:
        dictionary: Dictionary that maps movie titles to imdb links
    """
    dictionary = {}
    url = "https://www.imdb.com/title/tt"
    for x in range(len(ids)):
        zeros = 7 - len(str(ids[x]))
        movie_url = url + (zeros*"0") + str(ids[x])
        dictionary[titles[x]] = movie_url
    return dictionary

def generate_html(ids,titles):
    """
    Creates a html structure.

    Args:
        ids: List with Imdb ids
        titles: List with Movie Titles
        
    Returns:
        html_str: HTML markup code 
    """
    recommendations_dict = make_dict(ids,titles)
    html_str = "<ul style=\"list-style-type: none; padding: 0; margin: 0;\">"
    for movie, link in recommendations_dict.items():
        html_str += f"<li><h2><a href=\"{link}\" target=\"_blank\" style=\"display: block; text-align: center;\">{movie}</a></h2></li>"
    html_str += "</ul>" 
    return html_str

def get_recommendations(*args):
    answer_dict = {}
    similar_movies = []
    sim_movies = []
    imdb_ids = []
    similar_indices = []

    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = args[i]

    index = answer_dict["index"].upper()
    n_recommendations = answer_dict["n_recommendations"]
    movie_title = movie_finder(answer_dict["movie_title"])

    if movie_title == None:
        return "Movie Unrecognized. Try Again", None
    
    title = movie_title
    idx = movie_idx[title]

    if index == "FAISS":
        faiss_index = faiss.read_index(FAISS_PATH)
        query = embeddings[idx].reshape(1,-1)
        distances, indices = faiss_index.search(query,k=n_recommendations+1)
        similar_indices = indices[0].tolist()
    elif index == "ANNOY":
        dimension = annoy_config["dimension"]
        metric = annoy_config["metric"]
        annoy_index = AnnoyIndex(dimension,metric)
        annoy_index.load(ANNOY_PATH)
        similar_indices = annoy_index.get_nns_by_item(idx, n_recommendations+1)

    if idx in similar_indices:
        similar_indices.remove(idx)
    similar_indices=similar_indices[:n_recommendations]
    similar_movies = movies["title"].iloc[similar_indices]
    sim_imdb_ids = movies["imdbId"].iloc[similar_indices]

    for id in sim_imdb_ids:
        imdb_ids.append(id)
    for movie in similar_movies:
        sim_movies.append(movie)
    response = generate_html(imdb_ids,sim_movies)
    
    state ={
        "movie_idx":idx,
        "similar_indices":similar_indices,
        "movie_title":movie_title,
        "index":index
        }
    return response, state

with gr.Blocks() as demo:
    with gr.Tab("Movie Recommender"):
        with gr.Row():
            with gr.Column():
                index_input = gr.Radio(
                    label="Choose the index",
                    choices=["Faiss","Annoy"],
                    value="Faiss"
                )
                n_recommendations_input = gr.Slider(
                    label="Number of recommendations", 
                    minimum = 1,
                    maximum = 15,
                    step = 1,
                    randomize = True
                )
                movie_title_input = gr.Textbox(
                    lines = 1,
                    placeholder = "Enter a movie title",
                    label = "Choose a movie",

                )
                neighbor_state=gr.State()
                get_recommendation_btn = gr.Button(value = "Get Recommendations")
                get_recommendations_html = gr.HTML(label="Recommendations")
                get_recommendation_btn.click(
                    get_recommendations,
                    inputs=[
                        index_input,
                        n_recommendations_input,
                        movie_title_input
                        ],
                        outputs=[get_recommendations_html,neighbor_state],
                        api_name = "Recommender"
                )
    with gr.Tab("Plot Query and Neigbors"):
        plot = gr.Plot(label="Scatter Plot")
        submit_btn=gr.Button(value="Plot")
        submit_btn.click(
            get_plot,
            inputs=neighbor_state,
            outputs = plot
        )
        
demo.launch()