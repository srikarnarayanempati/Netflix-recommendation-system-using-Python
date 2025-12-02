from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

app = Flask(__name__)

warnings.filterwarnings('ignore')

org_movies = pd.read_csv('movie_dataset.csv')

# Handle NaN values by replacing them with an empty string
movies = org_movies[['genres', 'keywords', 'cast', 'title', 'director']].fillna('')

# Combine features and create a 'combined_features' column
movies['combined_features'] = movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['cast'] + ' ' + movies['title'] + ' ' + movies['director']

cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['combined_features'])
cs = cosine_similarity(count_matrix)

def get_movie_name_from_index(index):
    return org_movies[org_movies['index'] == index]['title'].values[0]

def get_index_from_movie_name(name):
    return org_movies[org_movies['title'] == name]['index'].values[0]

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    movie_index = get_index_from_movie_name(movie_name)
    similar_movies = list(enumerate(cs[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]

    recommended_movies = []
    for i in range(10):  # You can change the number of recommendations displayed
        recommended_movies.append(get_movie_name_from_index(sorted_similar_movies[i][0]))

    return render_template('page.html', movie_name=movie_name, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run()
