#collaborative filtering
import pandas as pd
from math import sqrt
import numpy as np

#reading the files
movie = pd.read_csv("movies.csv")
rating = pd.read_csv("ratings.csv")

#formatting the data
#creating a year column by extracting if from the title column
movie['year'] = movie.title.str.extract('(\\d\d\d\d\))',
expand=False)
movie['year'] = movie.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movie['title'] = movie.title.str.replace('(\(\d\d\d\d\))', '')
#getting rid of whitespace
movie['title'] = movie['title'].apply(lambda x: x.strip())
#getting rid of the genres column
movie.drop(columns=['genres'], inplace=True)

#profile of the user I am recommending for
user = [
            {'title':'Breakfast Club, The', 'rating':4},
            {'title':'Toy Story', 'rating':2.5},
            {'title':'Jumanji', 'rating':3},
            {'title':"Pulp Fiction", 'rating':4.5},
            {'title':'Akira', 'rating':5}
         ] 

#creates table with user values
inputMovie = pd.DataFrame(user)
Id = movie[movie['title'].isin(inputMovie['title'].tolist())]
inputMovie = pd.merge(Id, inputMovie)

#finding users who have watched the same movies
users = rating[rating['movieId'].isin(inputMovie['movieId'].tolist())]
userSubsetGroup = users.groupby(['userId'])

#sorted so users with most ratings in common are at the top
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)

#data table with user, movieId, rating
userSubsetGroup = userSubsetGroup[0:100]

#pearson coefficient gives you the strength of a linear relatiosnship from -1 to 1
#the closer to -1 means that the users are not similar at all
#the closer to 1 means that the users are similar to all

#key is id, val is pearson coefficient
pearson = {}
for name, group in userSubsetGroup:
    group = group.sort_values(by='movieId')
    inputMovie = inputMovie.sort_values(by='movieId')
    n = len(group)
    #scores for the movies that they both have in common
    temp = inputMovie[inputMovie['movieId'].isin(group['movieId'].tolist())]
    tempRatingList = temp['rating'].tolist()
    tempGroupList = group['rating'].tolist()
    #calculating pearson coefficient using the formula, x and y represent the two different users we are comparing
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(n) #
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(n)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(n)

    if Sxx != 0 and Syy != 0:
        pearson[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearson[name] = 0

#since keys are rows in the data the orientation is index
pearsonDF = pd.DataFrame.from_dict(pearson, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))

#sort users from highest to lowest based on pearson coefficient and take the top 50
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
#merge top users with rating to get ratings from top users
topUsersRating=topUsers.merge(rating, left_on='userId', right_on='userId', how='inner')
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']

recommendation_df = pd.DataFrame()
#weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
#sort from highest to lowest
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

print(movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())])