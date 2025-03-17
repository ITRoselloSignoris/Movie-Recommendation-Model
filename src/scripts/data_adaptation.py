import pandas as pd
from funpymodeling import status

MOVIES_PATH = "../datasets/raw_data/movies.csv"
RATINGS_PATH = "../datasets/raw_data/ratings.csv"
LINKS_PATH = "../datasets/raw_data/links.csv"

def extract_year(df):
    year_of_release = []
    title_corrected = []
    for title in df:
        if title.endswith(" "):
            title=title[:-1]
        title = title.split(" ")
        try:
            year= title[-1].strip("()")
            year = int(year) if len(year) == 4 else "nan"
        except:
            year = "nan"
        title.remove(title[-1])
        movie_title=""
        for x in range(len(title)):
            movie_title += title[x] + " "
        if movie_title.lower().endswith(",the "):
            axis=movie_title.lower().rfind(",the")
            movie_title = "The " + movie_title[:axis]
        elif movie_title.lower().endswith(", the "):
            axis=movie_title.lower().rfind(", the")
            movie_title = "The " + movie_title[:axis]
        movie_title = movie_title[:-1] if movie_title.endswith(" ") else movie_title
        title_corrected.append(movie_title)
        year_of_release.append(year)
    return title_corrected, year_of_release

def data_adaptation(moviesdf, ratingsdf, linksdf):
    moviesdf["title"], moviesdf["year"] = extract_year(moviesdf["title"])

    ratingsdf = pd.merge(ratingsdf,moviesdf[["title","movieId"]], on="movieId")
    linksdf = linksdf.drop("tmdbId",axis=1)
    data = pd.merge(ratingsdf,linksdf,on="movieId")
    data = data.drop("movieId",axis=1)
    data = data.drop("timestamp", axis = 1)

    genres = []
    for genre in moviesdf["genres"]:
        movieGenres = []
        m = genre.split("|")
        for x in m:
            movieGenres.append(x)
        genres.append(movieGenres)

    moviesdf["genres"] = genres
    moviesdf = pd.merge(moviesdf,linksdf,on="movieId")
    moviesdf = moviesdf.drop("movieId",axis=1)

    #Saving modified datasets
    data.to_csv("../datasets/adapted_data/adaptedRatingsData.csv", index = False)
    moviesdf.to_csv("../datasets/adapted_data/adaptedFeaturesData.csv", index = False)

if __name__ == "__main__":
    moviesdf = pd.read_csv(MOVIES_PATH, sep = ",")
    ratingsdf = pd.read_csv(RATINGS_PATH, sep = ",")
    linksdf = pd.read_csv(LINKS_PATH, sep = ",") 

    data_adaptation(moviesdf, ratingsdf, linksdf)