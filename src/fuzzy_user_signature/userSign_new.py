from contextlib import nullcontext
import pandas as pd
import numpy as np
from heapq import nlargest
from operator import itemgetter
from tqdm import tqdm


# This script is used to the construction of the User Signature matrices for every users. 
# From the user_matrix, the matrices representing the user's activities, with 1 in cell if 
# the user have seen a specific movie, than 0, we construct the User Signature, 
# i.e. a matrix in which, instead of the 1, there is the value of Tag Popularity. 
# The Tag Popularity is calculated for every tag (1-2-3-4-5) as 
# #number of resources tagged with that tag/ maximum numbero of tag used for a single resource.

# Function that load the dataset
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

    return data_users, items

def calc_film_attractivity (matrix):
    topic_attract = []
    max_film = 0

    for topic_column in matrix.columns:
        column = matrix[topic_column]
        # Get the count of non-Zeros values in column
        count_of_non_zeros = (column != 0).sum()
        topic_attract.append(count_of_non_zeros)
        if count_of_non_zeros > max_film:
            max_film = count_of_non_zeros

    for index in range(len(topic_attract)):
        topic_attract[index] = round (topic_attract[index]/max_film, 2)
    return topic_attract

def calc_tag_popularity (matrix):
    old_film_pop = []
    film_pop = []
    max_topic = 0

    old_film_pop = np.count_nonzero(matrix, axis=1)
    max_topic = nlargest(1, old_film_pop)[0]
    
    for index in range(len(old_film_pop)):
        new_element = round(old_film_pop[index]/max_topic,2)
        film_pop.append(new_element)

    return film_pop

def calc_user_signature (matrix, tag_popularity):
    user_signature = matrix * 0
    indices = pd.DataFrame(np.argwhere(matrix.gt(0).values)).groupby(0)[1].apply(list)
    j=0
    
    for ind in indices.index:
        tag_pop = tag_popularity[ind]
        list_non0 = indices.iloc[j]
        for elem in list_non0:
            new_elem_us = tag_pop
            user_signature.iat[ind,elem] = new_elem_us
        j+=1
    return user_signature

##### MAIN TO SAVE THE USER SIGNATURE FOR ALL USERS ####
data_users, items = loadData()

users = pd.read_csv('../ml-100k/u.user', sep="|", encoding='latin-1', header=None)
users.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']

for i in tqdm(users.index):
    topic_attract = 0
    film_pop = 0
    user_id = (users.iloc[i])['user id']

    filename = 'userSign' + str(user_id) + '.pkl'
    activity_user = pd.read_pickle('./matrices_new/' + filename)  
    activity_user = activity_user.loc[:,~activity_user.columns.duplicated()]
    #film_attract = calc_film_attractivity(activity_user)
    tag_pop = calc_tag_popularity(activity_user)
    user_signature = calc_user_signature(activity_user, tag_pop)
    filename_save = 'f_user_sign' + str(user_id) + '.pkl'
    print ('USER: ', user_id, " - FILE: ", filename_save)
    user_signature.to_pickle('./users_signature_new/' + filename_save)
