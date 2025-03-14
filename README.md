# Movie Recommendation Model 
## Project Overview
In this project I made a Movie Recommendation Model demo. The models were made with FAISS and ANNOY libraries.

### Simple Explanation on what the app does:
It gives you the option to choose between 2 models (FAISS and ANNOY), then you have to specify the ammount of recommendations you want, and last but not least, you write the  .

Once it, , it will show you a list of recommendations for the movie chosen, where each element recommended is a link to a page on IMDB dedcated to that element.
For the ones that do not know IMDB, it is a page 

Also, you have the possibility of plotting the scatter distribution of the movies, the movie chosen and the recommendations given in another tab.

### Purpose:
The purpose of this project was to try and experiment with different algorithms and tools that I did not know, to keep deepening in the differents possibilities that a library such as Gradio gives me, and to know how to "llevar a cabo" a recommendation model.

## Dataset
### Source Data 
The data used in this project came from https://grouplens.org/datasets/movielens/latest/. It contains . ratings across . movies. The data were created by . users.

This dataset was generated on July 20, 2023.

 4 files:
 - ratings.csv
 - tags.csv
 - movies.csv
 - links.csv

### Ratings Data File Structure (ratings.csv)

### Movies Data File Structure (movies.csv)

### Links Data File Structure (links.csv)



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
- Model Training (Collaborative Filtering, Content-Based Filtering, ANNOY and FAISS)
- Model deployment with an API and Docker.
- Graphical User Interface with Gradio.

## Technologies and Tools used
### <ins>1. Cleaning, Transformation and Data Preparation </ins> 
- **Pandas**: Load and prepare data, efficiently manipulate datasets, transform a dataset from JSON format to CSV format, perform descriptive analysis.

### <ins>2. Data Visualization / EDA</ins>
- **Funpymodeling**: Observe data distribution, the number of unique values and their occurrences, standard deviation, percentage of missing values, correlation between variables, among other information.
- **Seaborn y Matplotlib**:  Tools for data visualization and creation of statistical plots (Correlation, scatter plots, confusion matrix).

### <ins>3. Modeling</ins>
- **Scipy**: 
- **Scikit-Learn**
    - **TruncatedSVD and PCA**: Dimensionality Reduction ....
    - **NearestNeighbors**: 
    - **cosine_similarity**: 
- **Faiss**:
- **Annoy**:
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
├── deployment
│   ├── gradio_app
│   │   ├── dictionaries
│   │   │   ├── features_dense.pkl
│   │   │   ├── movie_idx.pkl
│   │   │   └── movies.pkl
│   │   ├── model
│   │   │   ├── annoy_config.json
│   │   │   ├── annoy_index.ann
│   │   │   └── faiss_index.bin
│   │   ├── app.py
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── mlruns   
│   ├── api.py
│   └── call_api.py
├── src
│   ├── datasets
│   │   ├── adapted_data
│   │   │   ├── adaptedFeaturesData.csv
│   │   │   └── adaptedRatingsData.csv
│   │   ├── processed_data
│   │   │   ├── featuresDataset.csv
│   │   │   └── processedData.csv
│   │   ├── raw_data
│   │   │   ├── links.csv
│   │   │   ├── movies.csv
│   │   │   ├── ratings.csv
│   │   │   └── tags.csv
│   ├── dictionaries
│   │   ├── features_dense.pkl
│   │   ├── movie_idx.pkl
│   │   └── movies.pkl
│   ├── model                                   
│   │   ├── annoy_config.json
│   │   ├── annoy_index.ann
│   │   └── faiss_index.bin
│   ├── notebooks
│   │   ├── 1. data_adaptation.ipynb
│   │   ├── 2. eda.ipynb
│   │   ├── 3. data_preparation.ipynb
│   │   └── 4. model_training.ipynb
│   ├── scripts
│   │   ├── data_adaptation.py
│   │   ├── data_preparation.py
│   │   └── training.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## To-Do list
- [ ] Improve the design for the recommendations given. 