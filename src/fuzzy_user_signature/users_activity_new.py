import pandas as pd
from tqdm import tqdm

#This Script is used for the creation of the user activities matrix. For every user we constuct a matrix with the movies on the columns
#and the ratings on the rows and in every cell, instead of 1, that is in the matrices_new, there is the minimum between the Tag Popularity and the Res Attractivity.

def loadData ():
    #MATRIX R: rating
    data = pd.read_csv('../ml-100k/u.data', sep='\\t', engine='python', names=['user id', 'movie id', 'rating', 'timestamp'])


    #MATRIX GENERAL: item and topic
    items = pd.read_csv('../ml-100k/u.item', sep="|", encoding='latin-1', header=None)
    items.columns = ['movie id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 
                    'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    items.drop(columns = ['release date', 'video release date', 'IMDb URL'], axis=1)

    #MATRIX F+R
    data_users = pd.merge(data[['user id', 'movie id','rating' ]], items, on='movie id')
    #data_users.to_csv('ml-100k/test.csv', index=False)

    #LIST OF GENRES
    genres = pd.read_csv('../ml-100k/u.genre', sep="|", encoding='latin-1', header=None)
    genres.drop(genres.columns[1], axis=1, inplace=True)
    genres.columns = ['Genres']
    genre_list = list(genres['Genres'])
    genre_list.remove("unknown")
    return data_users, genre_list, items

def user_act_2 (movies, user_activity, user_sign):
    j=0

    for movie in movies:
        rating = user_activity.loc[user_activity['movie_title'] == movie]['rating']
        if not(rating.empty):
            rate = rating.values[0]
            user_sign.iat[rate-1, j] = 1
        j+=1

    return user_sign

##### MAIN TO SAVE THE TABLE FOR ALL USERS ####
data_users, genres, items = loadData()
activity_matrices = []
items = items.drop(columns = ['movie id', 'release date', 'video release date', 'IMDb URL'], axis=1)
movies = items['movie_title'].values

users = pd.read_csv('../ml-100k/u.user', sep="|", encoding='latin-1', header=None)
users.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']

for i in tqdm(users.index-1):
    user_signature = pd.DataFrame(0, columns=movies, index = [1,2,3,4,5])
    user_id = (users.iloc[i])['user id']
    user_activity = data_users.loc[data_users['user id'] == user_id]
    user_activity = user_activity.reset_index()
    user_signature = user_act_2 (movies, user_activity, user_signature)
    print ('USER: ', user_id)
    filename = 'userSign' + str(user_id) + '.pkl' 
    print (filename)
    user_signature.to_pickle('./matrices_new/'+ filename)


