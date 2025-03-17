# Movie Recommendation Model 
## Project Overview
In this project I made a Movie Recommendation Model demo. The indexes were made with FAISS and ANNOY libraries.

It gives you the option to choose between two indexes (FAISS and ANNOY), then you have to specify the number of recommendations you want, and last but not least, you enter the movie for which you are seeking recommendations.

Once you entered those parameters, it will show you a list of recommendations for the movie chosen, where each element recommended is a link to a page on IMDB dedicated to that element.
For the ones that do not know IMDB, it is an online database with information about movies, tv-series, podcasts, among other type of audivisual content.

Also, you have the possibility of plotting the scatter distribution of the movies, the movie chosen and the recommendations given in another tab. 

The purpose of this project was to try and experiment with different algorithms and tools that I did not know, to keep deepening in the differents possibilities that a library such as Gradio gives me, and to know how to carry out a recommendation model project.

## Dataset
### Source Data 
The data used in this project came from https://grouplens.org/datasets/movielens/latest/. It contains 100836 ratings across to 9742 movies. The data were created by 610 users.

This dataset was generated on September 26, 2018.

<ins>It had 5 files, however I only needed the following 3 files:</ins>
 - **ratings.csv**
 - **movies.csv**
 - **links.csv**

#### Ratings Data File Structure (ratings.csv)
All ***ratings*** are contained in this file. Each line of this file after the header row represents one rating of one movie by one user, and has the following columns:
`userId`,`movieId`,`rating` and `timestamp`

*Ratings* are made on a 5-star scale, with half-star increments (0.5 stars -5.0 stars)

*Timestamps* represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.
#### Movies Data File Structure (movies.csv)
***Movie information*** is contained in this file. Each line of this file after the header row represents one movie, and has the following columns:
`movieId`,`title` and `genres`

*Movie titles* include the year of release in parentheses.

*Genres* are a pipe-separated list.

#### Links Data File Structure (links.csv)
***Identifiers*** that can be used to link to other sources of movie data are contained in this file. Each line of this file after the header row represents one movie, and has the following columns:
`movieId`, `imdbId` and `tmdbId`

`movieId` is an identifier for movies used by https://movielens.org/.

`imdbId` is an identifier for movies used by http://www.imdb.com/.

`tmdbId` is an identifier for movies used by https://www.themoviedb.org/.

### Data Preprocessing 
I added a new column called `"year"`, because the year of release for each movie was in the same string as the movie title. As I thought it would be better to turn this years into decades, I splitted the years from the movie\`s titles. Regarding the movie\`s titles I fixed some mispelled titles.

I replaced the *movielens* Ids for the *imdb* Ids so I could use them later for the app.

Next, the genres for each movie were written in a string value. So I transformed the genres from the previous format (*"Action|Comedy"*) into a list where each element of the list is a genre (*["Action","Comedy"]*)

Then, I created a new column called `"decade"` and replaced each value in the `"year"` column with its corresponding decade, for example, I transformed 1967 into 1960s. 

Moreover, I applied **One Hot Encoding** in the `"genres"` column and in the `"decade"` column. 

Finally, I saved both datasets, one for ***Collaborative Filtering*** and another one for ***Content-Based Filtering***.

## Methods used
- Data Cleaning and Transformation (Data Wrangling)
- Exploratory Data Analysis (EDA)
- Data Visualization.
- Data Preparation.
- Model Training (Collaborative Filtering, Content-Based Filtering, KNN)
- Index Training (Faiss, Annoy, )
- Model deployment with an API and Docker.
- Graphical User Interface with Gradio.

## Technologies and Tools used
### <ins>1. Cleaning, Transformation and Data Preparation </ins> 
- **Pandas**: Load and prepare data, efficiently manipulate datasets, transform a dataset from JSON format to CSV format, perform descriptive analysis.

### <ins>2. Data Visualization / EDA</ins>
- **Funpymodeling**: Observe data distribution, the number of unique values and their occurrences, standard deviation, percentage of missing values, correlation between variables, among other information.
- **Seaborn y Matplotlib**:  Tools for data visualization and creation of statistical plots (Correlation, scatter plots, confusion matrix).

### <ins>3. Modeling</ins>
- **Scipy**: For scientific and technical computing. In this case in order to make the sparse matrix.
- **Scikit-Learn**
    - **TruncatedSVD and PCA**: Dimensionality Reduction Techniques.
    - **NearestNeighbors**: Used to identify the closest data points to a given query point.
    - **cosine_similarity**: Quantifies similarity between vectors.
- **Faiss**: Built to scale similarity search to very large datasets and high-dimensional vectors.
- **Annoy**: Designed for efficient, memory-light approximate nearest neighbor (ANN) search. 
- **Mlflow**: Management and registration of model experiments, allowing us to track parameters, metrics and versions.

### <ins>4. API</ins>
- **FastAPI**: API creation in order to expose the trained model.
- **Uvicorn**: Light and fast ASGI server that allows the asyncronous execution of the API.
- **Requests**: Make HTTP requests to the API.
- **Pydantic**: Validate the input data of the API.
- **Gradio**: Graphical User Interface that allows the users to test the model.

### <ins>5. Deployment</ins> 
- **Docker**: Containerize the API and simplify its deployment.
- **Hugging Face Space**: Host and share the model  in a centralized way.

### <ins>Programming Language</ins>
- **Python**: Main language used for the development of the project, compatible with the libraries of data science and machine learning.

## Installation and Setup
### Docker
1. Build image:  
`docker build -t movie_recommendation_model .`

2. Build container:  
`docker run -p 7860:7860 -e ID_USER=Iñaki movie_recommendation_model` 


## Code Structure
```bash
├── deployment            # Folder MLops deployment files (API, app, mlflow)
│   ├── gradio_app          # Folder with necessary gradio app files (Deployed in HF Spaces)
│   │   ├── dictionaries                    
│   │   │   ├── features_dense.pkl          
│   │   │   ├── movie_idx.pkl               
│   │   │   └── movies.pkl                  
│   │   ├── index                        
│   │   │   ├── annoy_config.json           
│   │   │   ├── annoy_index.ann             
│   │   │   └── faiss_index.bin             
│   │   ├── app.py                # Gradio app script
│   │   ├── README.md             # README.md file required by HuggingFace Spaces
│   │   └── requirements.txt      
│   ├── mlruns                              # Run metadata for mlflow 
│   ├── api.py                              # Api python script
│   └── call_api.py                         
├── src                                     # Src folder 
│   ├── datasets                            # Datasets Folder
│   │   ├── adapted_data                    # Adapted Data
│   │   │   ├── adaptedFeaturesData.csv         # Movies Features Dataset
│   │   │   └── adaptedRatingsData.csv          # Movies Ratings Dataset
│   │   ├── processed_data                  # Cleaned Data
│   │   │   ├── featuresDataset.csv             # Movies Features Dataset
│   │   │   └── processedData.csv               # Movies Ratings Dataset
│   │   ├── raw_data                        # Unprocessed datasets
│   │   │   ├── links.csv                   
│   │   │   ├── movies.csv                  
│   │   │   └── ratings.csv                 
│   ├── dictionaries                        # Dictionaries required 
│   │   ├── features_dense.pkl              # Features df with PCA applied
│   │   ├── movie_idx.pkl                   # Maps title to idx in features dense
│   │   └── movies.pkl                      # Movie titles and imdb Ids
│   ├── index                             # Indexes folder
│   │   ├── annoy_config.json               # ANNOY training parameters 
│   │   ├── annoy_index.ann                 # ANNOY index
│   │   └── faiss_index.bin                 # FAISS index
│   ├── notebooks                           # Jupyter Notebooks
│   │   ├── 1. data_adaptation.ipynb        
│   │   ├── 2. eda.ipynb                    
│   │   ├── 3. data_preparation.ipynb       
│   │   └── 4. model_training.ipynb         
│   ├── scripts                             # Python Scripts
│   │   ├── data_adaptation.py              
│   │   ├── data_preparation.py             
│   │   └── training.py                     
├── .dockerignore                           
├── Dockerfile                              
├── README.md                               # Project overview
└── requirements.txt                        # Libraries required to run the project
```

## To-Do list
- [ ] Improve the design for the recommendations given. 