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

def get_movie_recommendations(user_movie, data, similarity_matrix, popularity_scores, recommendation_type='combined'):
    list_movies = data['title'].tolist()
    close_matches = difflib.get_close_matches(user_movie, list_movies)
    
    if not close_matches:
        print("No close matches found. Please try another movie.")
        return
    
    close_match = close_matches[0]
    print(f"Closest match found: {close_match}")

    movie_index = data[data['title'] == close_match].index.values[0]

    if recommendation_type == 'similarity':
        scores = list(enumerate(similarity_matrix[movie_index]))
    elif recommendation_type == 'popularity':
        scores = list(enumerate(popularity_scores))
    else:
        print("Invalid recommendation type. Please choose 'similarity' or 'popularity'.")
        return

    sorted_movies = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"Top 10 {recommendation_type.capitalize()} movies for you:")
    for i, movie in enumerate(sorted_movies[:10], 1):
        index, score = movie
        index_title = data.loc[data.index == index, 'title'].values[0]
        print(f"{i}. {index_title} ({recommendation_type.capitalize()} Score: {score:.2f})")

def popularity_based_recommendations(data):
    # Simple popularity based on the number of ratings
    popularity_scores = data.groupby('title')['vote_count'].sum().sort_values(ascending=False)
    popularity_scores = popularity_scores.reset_index().rename(columns={'vote_count': 'popularity_score'})
    
    return popularity_scores['popularity_score']

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

    # Popularity based on the number of votes
    popularity_scores = popularity_based_recommendations(movies_data)

    # User input
    movie_name = input("Enter your favorite movie name: ")
     
    
    type_input = int(input("Enter 1 for similarity, or 2 for popularity: "))

    if type_input == 1:
     # Get movie recommendations based on similarity
     print("Similarity recommendation")
     get_movie_recommendations(movie_name, movies_data, similarity_matrix, popularity_scores, recommendation_type='similarity')

    elif type_input == 2:
       # Get movie recommendations based on popularity
       print("Popularity recommendation")
       get_movie_recommendations(movie_name, movies_data, similarity_matrix, popularity_scores, recommendation_type='popularity')

    else:
      print("Invalid input. Please enter 1 or 2.")