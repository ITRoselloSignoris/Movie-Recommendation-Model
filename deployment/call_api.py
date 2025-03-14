import requests

data1 = {
    "model":"faiss",
    "n_recommendations":8,
    "movie_title": "Rocky"
}


data2 = {
    "model":"ANNOY",
    "n_recommendations":4,
    "movie_title": "Star Wars: Episode IV - A New Hope"
}


response = requests.post("http://0.0.0.0:7860/recommendation",json=data1)
print(response.json())

response = requests.post("http://0.0.0.0:7860/recommendation",json=data2)
print(response.json())