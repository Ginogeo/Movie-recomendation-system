import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_data(data):
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        data[feature] = data[feature].fillna('')
    data['combined_features'] = data[selected_features].agg(' '.join, axis=1)
    return data

def get_movie_recommendations(user_movie, data, similarity_matrix):
    list_movies = data['title'].tolist()
    close_matches = difflib.get_close_matches(user_movie, list_movies)
    
    if not close_matches:
        print("No close matches found. Please try another movie.")
        return
    
    close_match = close_matches[0]
    print(f"Closest match found: {close_match}")

    movie_index = data[data['title'] == close_match].index.values[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print("Top 20 movies for you:")
    for i, movie in enumerate(sorted_movies[:10], 1):
        index, score = movie
        index_title = data.loc[data.index == index, 'title'].values[0]
        print(f"{i}. {index_title} (Similarity Score: {score:.2f})")

if __name__ == "__main__":
    # Loading data
    movies_data = pd.read_csv('C:/Users/MY BOOK/Downloads/movies.csv')

    # Data preprocessing
    movies_data = preprocess_data(movies_data)

    # Converting text data to numerical data
    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(movies_data['combined_features'])

    # Finding similarity score
    similarity_matrix = cosine_similarity(feature_vector)

    # User input
    movie_name = input("Enter your favorite movie name: ")

    # Get movie recommendations
    get_movie_recommendations(movie_name, movies_data, similarity_matrix)
