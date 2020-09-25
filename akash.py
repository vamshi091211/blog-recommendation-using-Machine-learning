# IMPORTING LIBRARIES

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LOADING DATA

user_info = pd.read_csv("user.csv", engine = "python")
user_and_post_info = pd.read_csv("view.csv")
post_info = pd.read_csv("post.csv")

# removing anomalies
#post_info.drop(['Unnamed: '+str(x) for x in range(4, 10)], axis = 1, inplace = True)


# GETTING IDs

user_ids = pd.DataFrame(user_info['_id'])
post_ids = pd.DataFrame(post_info['_id'])

#

post_info.insert(0, 'index', range(0, len(post_info)))
post_info['_id'] = post_info['_id'].astype(str)

features = ['title', 'category', ' post_type']

for feature in features:
    post_info[feature] = post_info[feature].fillna("")
    
def combine_features(row):
    try:
        return row['title'] + " " + row['category'] + " " + row[' post_type']
    except:
        print ("Error:", row)
        
post_info["combined_features"] = post_info.apply(combine_features, axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(post_info["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

def get_index_from_title(title):
    return post_info[post_info['title'] == title]['index'].values[0]

def get_title_from_index(index):
    return post_info[post_info['index'] == index]['title'].values[0]

def get_post_id_from_user_id(userid):
    return user_and_post_info[user_and_post_info['user_id'] == userid]['post_id'].values[0]

def get_title_from_post_id(postid):
    return post_info[post_info['_id'] == postid]['title'].values[0]
    
#post_user_likes='MIS'

Id_of_user = '5e5dfbbefbc8805f69e02c91'
Id_of_post_user_likes = get_post_id_from_user_id(Id_of_user)
post_user_likes = get_title_from_post_id(Id_of_post_user_likes)
#print(post_user_likes)
post_index = get_index_from_title(post_user_likes)

similar_posts = list(enumerate(cosine_sim[post_index]))
sorted_similar_posts = sorted(similar_posts,key=lambda x:x[1],reverse=True)

i=0
for post in sorted_similar_posts:
    print(get_title_from_index(post[0]))
    i=i+1
    if i>50:
        break
