import pandas as pd
from tqdm import tqdm
from heapq import nlargest
from operator import itemgetter
import warnings
import numpy as np
import sys

# This script calculate the similarities of every user. This is done by the user kindredness: we calculated the minimum matrix
# of the two users' matrices of that we want to calculate the similarities and the kindredness with a specific formula. The ouput is a file F-neihbors
# that contains a dictionary with the keys the users IDs and the values the list of tuples [(neighbor, kindredness)] for each user. These files are
# stored in the folder 'Make_Prediction' beacuse are used for the calculation of the predictions. This file is produced 10 times, for every training fold of the
# 10-cross-fold validation.

def load_filter_data(num_train):
        #MATRIX R: rating
    
    data_users = pd.read_pickle('training' + str(num_train) + '.pkl')
    users = pd.read_pickle('users.pkl')

    #MATRIX F: item and topic
    items = pd.read_csv('../ml-100k/u.item', sep="|", encoding='latin-1', header=None)
    items.columns = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 
                    'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    items.drop('release date', axis=1, inplace=True)
    items.drop('video release date', axis=1, inplace=True)
    items.drop('IMDb URL', axis=1, inplace=True)

    #LIST OF GENRES
    genres = pd.read_csv('../ml-100k/u.genre', sep="|", encoding='latin-1', header=None)
    genres.drop(genres.columns[1], axis=1, inplace=True)
    genres.columns = ['Genres']
    genre_list = list(genres['Genres'])

    return data_users, users, genre_list
    
def calc_similarity (user, neigh):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    similarity = 0
    summ = 0


    t_norm_matrix = pd.concat([user,neigh]).min(level=0)
    summ = t_norm_matrix.values.sum()

    sum_user = user.values.sum()

    similarity = round(summ/sum_user, 2)
    return similarity


neighborhood = 4
data_neigh = []
i=0

num_train = sys.argv[1]

data_users, users, genres = load_filter_data(num_train)
print (data_users)
print (users)

for i in tqdm(users.index-1):
    userDict = {}
    neighbors = []
    similarities = []
    user_interest_id = (users.iloc[i])['user id']
    print ("##### USER: ", user_interest_id)
    filename_user = 'f_user_sign' + str(user_interest_id) + '.pkl'
    user_interest = pd.read_pickle('./users_signature_new/' + filename_user)  
    print (user_interest)
    users_compare = users.loc[users['user id']!= user_interest_id]
    print (users_compare)
    for j in (users_compare.index-1):
        new_user_id = users_compare.iloc[j]['user id']
        filename_neigh = 'f_user_sign' + str(new_user_id) + '.pkl'
        neighbor = pd.read_pickle('./users_signature_new/' + filename_neigh)  
        similarities.append([new_user_id, calc_similarity (user_interest, neighbor)])
    neighbors = nlargest(15, similarities, key=itemgetter(1))

    userDict = {'user': int(user_interest_id), 'neighbors' : neighbors}
    print(userDict)
    data_neigh.append(userDict)

    usersData = pd.DataFrame.from_records(data_neigh)
    usersData.to_pickle("F-neighbors-" + str(num_train)+ ".pkl")