
import pandas as pd
import sys

filename = sys.argv[1]
path = '../ml-100k/' + sys.argv[1]
num_users = 497
i = 0
j = 0
flag_user = False
flag_item = False
ind = 0
us_list = []

#MATRIX R: rating
data = pd.read_csv(path, sep='\\t', engine='python', names=['user id', 'movie id', 'rating', 'timestamp'])

#MATRIX F: item and topic
items = pd.read_csv('../ml-100k/u.item', sep="|", encoding='latin-1', header=None)
items.columns = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 
                'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items.drop('release date', axis=1, inplace=True)
items.drop('video release date', axis=1, inplace=True)
items.drop('IMDb URL', axis=1, inplace=True)

#MATRIX F+R
data_users = pd.merge(data[['user id', 'movie id','rating' ]], items, on='movie id')

#MATRIX OF USERS
users = pd.read_csv('../ml-100k/u.user', sep="|", encoding='latin-1', header=None)
users.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']

#Extraction of top-n users, i.e. the users with the highest number of ratings
group_id = data_users.groupby("user id").size().to_frame('size')
n_larg = group_id['size'].nlargest(num_users)
topn_user_list = n_larg.index.values

#Filter the old data and the old users matrices with only the top-n users extracted before
data_users = data_users[data_users['user id'].isin(topn_user_list)]
users = (users[users['user id'].isin(topn_user_list)]).reset_index(drop=True)

#Delete from the dataset the movies rated less than 20 times
while flag_item == False:
    flag_item = True

    for j in (items.index):
        movie_id = items.iloc[j]['movie id']
        if len(data_users.loc[data_users['movie id'] == movie_id])<20 and (not((data_users.loc[data_users['movie id'] == movie_id]).empty)):
            data_users = data_users.loc[data_users['movie id'] != movie_id]
            flag_item = False

    i+=1

data_users_final = data_users.reset_index(drop=True)

#Calculate the percentage
percent = lambda part, whole:float(whole) / 100 * float(part)
init = 0
final = 0

###### 10-CROSS-FOLD VALIDATION ######
# This loop divide the original dataset in 90% training and 10% test for 10 times (folds). 
# Every time is taken a part of dataset different than the previoud one
for i in range(10):
    print ("####### ITERAZIONE: ", i)
    test_set = training_set = pd.DataFrame(columns = ['user id', 'movie id', 'rating', 'movie title', 'unknown', 'Action', 
                    'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    for ind in users.index:
        user_id = users.iloc[ind]['user id']
        user_data = (data_users_final.loc[data_users_final['user id'] == user_id]).reset_index(drop=True)
        len_user_data = len(user_data)
        perc = round(percent(10, len_user_data),0)
        init = perc*i
        final = perc*(i+1)
        user_test = user_data.loc[init:final-1]
        user_train = user_data.drop(user_test.index)
        test_set = pd.concat([test_set, user_test])
        training_set = pd.concat([training_set, user_train])

    
    
    #SAVING TO FILE PKL TRAIN AND TEST FOR EVERY FOLD
    training_set = training_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)
    training_set.to_pickle ('../Fuzzy_Method/' + 'training' + str(i) + '.pkl')
    test_set.to_pickle ('../Make_Prediction/test/' + 'test' + str(i) + '.pkl')
    print ("######### TRAINING SET\n", training_set)
    print ("########## TEST SET\n", test_set)

#SAVING THE TOP-N USERS
users.to_pickle('users.pkl')
