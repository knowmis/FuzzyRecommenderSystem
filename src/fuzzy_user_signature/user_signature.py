import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_matrix_ones(movies, user_activity):
    user_sign = pd.DataFrame(0, index=[1, 2, 3, 4, 5], columns=movies)
    for movie in movies:
        if movie in user_activity['movie_title'].values:
            rate = user_activity.loc[user_activity['movie_title'] == movie, 'rating'].iloc[0]
            user_sign.at[rate, movie] = 1
    return user_sign

def calc_film_attractivity(matrix):
    topic_attract = (matrix != 0).sum(axis=1) / (matrix != 0).sum(axis=1).max()
    return topic_attract.round(2)

def calc_tag_popularity(matrix):
    film_pop = np.count_nonzero(matrix, axis=0) / np.count_nonzero(matrix, axis=0).max()
    return film_pop.round(2)

def calc_user_signature(matrix, tag_popularity):
    user_signature = matrix.multiply(tag_popularity, axis="columns")
    return user_signature

for i in range(10):
    data_path = f'../../data/training_data/training{i}.pkl'
    data_users = pd.read_pickle(data_path)
    users_path = '../../data/user_data/users.pkl'
    users = pd.read_pickle(users_path)

    items_path = '../../data/item_data/items.pkl'
    items = pd.read_pickle(items_path)
    movies = items['movie_title']

    for user_id in tqdm(users['user_id']):
        user_activity = data_users[data_users['user_id'] == user_id]

        if not user_activity.empty:
            user_signature = create_matrix_ones(movies, user_activity)
            tag_pop = calc_tag_popularity(user_signature)
            user_signature = calc_user_signature(user_signature, tag_pop)

            save_path = f'../../data/user_signatures/fold{i}'
            os.makedirs(save_path, exist_ok=True)
            filename_save = f'f_user_sign{user_id}.pkl'
            user_signature.to_pickle(os.path.join(save_path, filename_save))
