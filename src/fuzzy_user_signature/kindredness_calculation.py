import pandas as pd
from tqdm import tqdm
from heapq import nlargest
from operator import itemgetter
import numpy as np
import sys

def load_filter_data(num_train):
    data_users = pd.read_pickle(f'../../data/training_data/training{num_train}.pkl')
    users = pd.read_pickle('../../data/user_data/users.pkl')
    return data_users, users

def calc_similarity(user, neighbor):
    user_aligned, neighbor_aligned = user.align(neighbor, join='inner', axis=1)
    user_array = user_aligned.values
    neighbor_array = neighbor_aligned.values
    t_norm_array = np.minimum(user_array, neighbor_array)
    summ = np.sum(t_norm_array)
    sum_user = np.sum(user_array)
    return round(summ / sum_user, 2)

def main(num_train):
    data_users, users = load_filter_data(num_train)
    data_neigh = []

    for i in tqdm(range(len(users))):  # Usa range(len(users)) per iterare sugli indici
        user_id = users.iloc[i]['user_id']
        user_interest = pd.read_pickle(f'../../data/user_signatures/fold{num_train}/f_user_sign{user_id}.pkl')

        similarities = []
        for j in range(len(users)):  # Stesso uso di range(len(users)) per iterare
            if i != j:
                new_user_id = users.iloc[j]['user_id']
                neighbor = pd.read_pickle(f'../../data/user_signatures/fold{num_train}/f_user_sign{new_user_id}.pkl')
                similarity = calc_similarity(user_interest, neighbor)
                similarities.append((new_user_id, similarity))

        neighbors = nlargest(20, similarities, key=itemgetter(1))
        data_neigh.append({'user': user_id, 'neighbors': neighbors})

    usersData = pd.DataFrame(data_neigh)
    usersData.to_pickle(f"../../data/neighbors_data/F-neighbors-{num_train}.pkl")

if __name__ == "__main__":
    for i in range(10):  # Itera su ogni fold
        main(i)
