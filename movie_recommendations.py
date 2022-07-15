#content based learning
import pandas as pd
from ast import literal_eval
import numpy as np
#counts the frequency of each words and returns it as a 2d vector
from sklearn.feature_extraction.text import CountVectorizer
#takes the dot product of the vector, this shows distance, the greater the distance between vectors the less similar they are
from sklearn.metrics.pairwise import cosine_similarity
#content-based recommendation system

#gets the movies and credits from the data files
credits_df = pd.read_csv("tmdb_5000_credits.csv")
movies_df = pd.read_csv("tmdb_5000_movies.csv")

#merges the files to include credit id in movies_df
credits_df.columns = ['id','title','cast','crew']
movies_df = movies_df.merge(credits_df, on="id")

#literal eval find the features and makes the movies_df array only contain the features we need
features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

#gets name of the director
def get_director(d):
    for i in d:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

#gets top three names of the list
def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

movies_df["director"] = movies_df["crew"].apply(get_director)
features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)

#removes spaces and makes letters lowercase in data
def clean_data(row):
    if isinstance(row, list):
        return [str.lower(i.replace(" ", "")) for i in row]
    else:
        if isinstance(row, str):
            return str.lower(row.replace(" ", ""))
        else:
            return ""

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)

#String containing all metadata info
def create_soup(features):
    return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['director'] + ' ' + ' '.join(features['genres'])

movies_df["soup"] = movies_df.apply(create_soup, axis=1)

#converts "soup" into a vector
count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])

#gets the distance of each vector to compute similarity
cos_sim = cosine_similarity(count_matrix, count_matrix)
movies_df = movies_df.reset_index()

#idk what this is
indices = pd.Series(movies_df.index, index=movies_df['original_title'])
indices = pd.Series(movies_df.index, index=movies_df["original_title"]).drop_duplicates()

def get_recommendations(title, cosine_sim):
    index = indices[title]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores= sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    movies_indices = [i[0] for i in similarity_scores]
    movies = movies_df["original_title"].iloc[movies_indices]
    return movies
    
print("Recommendations for Despicable Me")
print(get_recommendations("Despicable Me", cos_sim))
print()
print("Recommendations for Avengers")
print(get_recommendations("The Avengers", cos_sim))