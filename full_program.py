class User:
    
    def __init__(self, user_id):
        self.userid = user_id
        self.interests = []
        self.visited = [0]*500
        self.nextrecomm = [None]*12
        self.lastpost = None
        self.stop = False
        
    def findInterests(self, unique_tags):
        k = 0
        for i in range(79):
            for j in range(3):
                if(k < len(unique_tags)):
                    print(unique_tags[k].center(40), end = "|")
                k+=1
            print("\n")
            
        print("choose tags as per your interest")
        while(True):
            i = input()
            if(len(i) == 0):
                break
            self.interests.append(i)
            
    def showRecomm(self):

        for i in range(len(self.nextrecomm)):
            '''
            if (i == 0):
                print("\nKNOWLEDGE BASED\n")
            elif(i == 3):
                print("\nCONTENT BASED\n")
            elif(i == 7):
                print("\nCOLLABRATIVE BASED\n")
            '''
            if (self.nextrecomm[i] != None):
                print(self.nextrecomm[i][1], self.nextrecomm[i][0])
            
    def selectPost(self):
        recomindex = []
        for x in self.nextrecomm:
            if x != None:
                recomindex.append(x[1])
            
        print("please select a post number from the list")
        n = int(input())
        if n == -1:
            print("Thank You")
            self.stop = True
            return -1
        if n not in recomindex:
            print("select again")
            self.selectPost()
            return 0
        
        self.lastpost = n
        self.visited[n] = 1
        for i in range(len(self.nextrecomm)):
            if self.nextrecomm[i] != None:
                if (self.nextrecomm[i][1] == n):
                    self.nextrecomm[i] = None
                    break
        print("\n")
        
    def resetrecomm(self):
        self.nextrecomm = [None]*10
        
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_kernels

user_info = pd.read_csv("user.csv", engine = "python")
user_and_post_info = pd.read_csv("view.csv")
post_info = pd.read_csv("post.csv")

# Adding extra index column (to be used later)
post_info.insert(0, 'index', range(0, len(post_info)))
# Converting ids to string data type
post_info['_id'] = post_info['_id'].astype(str)
# Removing NaN columns (Unnamed)
post_info.drop(['Unnamed: '+str(x) for x in range(4, 10)], axis = 1, inplace = True)

for i in range(len(post_info['category'])):
    temp = post_info['category'][i]
    if(type(temp) == float):
        post_info['category'][i] = "Project"
    temp = post_info['category'][i].replace("; ","|")
    temp = post_info['category'][i].replace(";","|")
    if(len(temp) != 0):
        j = 0
        while(temp[j] == " "):
            temp = temp[1:]
        post_info['category'][i] = temp

unique_tags = []
for x in post_info['category']:
    x = x.lower()
    tags = x.split("|")
    unique_tags += tags
    
unique_tags = list(set(unique_tags))

## MAKING BAG OF WORDS
post_info_cols = list(post_info.columns)
post_info_cols = post_info_cols[2:]
for feature in post_info_cols:
    post_info[feature] = post_info[feature].fillna("")
def combine_features(row):
    try:
        return row['title'] + " " + row['category'] + " "+ row[' post_type']
    except:
        print ("Error:",row)
# New Column having concatenated data of all columns (except '_id')
post_info["combined_features"] = post_info.apply(combine_features, axis = 1)
post_info["combined_features"]

## Finding Similarity Between Post Using Bag of Words (For Content Based Filtering)
cv = CountVectorizer()
count_matrix = cv.fit_transform(post_info["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

# FOR COLLABRATIVE FILTERING
user_info.iloc[67]["gender"] = "female"
user_info.iloc[67]["academics"] = "undergraduate"
user_info.iloc[74]["gender"] = "female"
user_info.iloc[74]["academics"] = "undergraduate"

from collections import defaultdict
user_info["person_type"] = user_info["gender"] + " " + user_info["academics"]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
user_info["person_type"] = le.fit_transform(user_info["person_type"])

post_likes = defaultdict(list)
for i in range(len(user_and_post_info)):
    if post_likes[user_and_post_info.iloc[i]["post_id"]] == []:
        post_likes[user_and_post_info.iloc[i]["post_id"]] = [0, 0, 0, 0]
    j = user_info[user_info["_id"] == user_and_post_info.iloc[i]["user_id"]]["person_type"]
    post_likes[user_and_post_info.iloc[i]["post_id"]][int(j)] += 1
    
total_likes = []
likes_0 = []
likes_1 = []
likes_2 = []
likes_3 = []
for key, value in post_likes.items():
    total_likes.append([sum(value), key])
    likes_0.append([value[0], key])
    likes_1.append([value[1], key])
    likes_2.append([value[2], key])
    likes_3.append([value[3], key])

total_likes.sort(reverse = True)
likes_0.sort(reverse = True)
likes_1.sort(reverse = True)
likes_2.sort(reverse = True)
likes_3.sort(reverse = True)
likes_all = {0:likes_0, 1:likes_1, 2:likes_2, 3:likes_3}

# Defining Utility Functions
def get_index_from_title(title):
    return post_info[post_info['title'] == title]['index'].values[0]

def get_title_from_index(index):
    return post_info[post_info['index'] == index]['title'].values[0]

def get_post_id_from_user_id(userid):
    return user_and_post_info[user_and_post_info['user_id'] == userid]['post_id'].values[0]

def get_title_from_post_id(postid):
    return post_info[post_info['_id'] == postid]['title'].values[0]

def get_category_from_title(title):
    return merged_df[merged_df['title']==title]['new_category'].values[0]

def knowledgeBasedRecommendation(user):
    target = " ".join(user.interests)
    candidates = list(post_info["category"])
    vec = CountVectorizer()
    vec.fit(candidates)
    cos_sim = pairwise_kernels(vec.transform([target]), vec.transform(candidates), metric = "cosine")
    similarity_pair = []
    for i in range(len(cos_sim[0])):
        similarity_pair.append((cos_sim[0][i], i))
    similarity_pair.sort(reverse = True)
    
    for i in range(0, 3):
        if user.nextrecomm[i] == None:
            for first, second in similarity_pair:
                if (get_title_from_index(second), second) not in user.nextrecomm and user.visited[second] == 0:
                    user.nextrecomm[i] = (get_title_from_index(second), second)
                    break
        
        
def contentBasedRecommendation(user):
    for i in range(3, 7):
        user.nextrecomm[i] = None
    post_index = user.lastpost
    # Passing the post_index to cosine_sim to make a list of similar posts
    similar_posts = list(enumerate(cosine_sim[post_index]))
    # Sorting the list in decreasing order of their similarity
    sorted_similar_posts = sorted(similar_posts, key = lambda x:x[1], reverse=True)
    temp = sorted_similar_posts.pop(0)
    for i in range(3, 7):
        if (user.nextrecomm[i] == None):
            for post in sorted_similar_posts:
                if (get_title_from_index(post[0]), post[0]) not in user.nextrecomm and user.visited[post[0]] == 0:
                    user.nextrecomm[i] = (get_title_from_index(post[0]), post[0])
                    break
        
            
def collabrativeRecommendation(user):
    user_id = user.userid
    p_type = user_info[user_info["_id"] == user_id]["person_type"]
    lst = likes_all[int(p_type)]
    
    for i in range(7, 10):
        if (user.nextrecomm[i] == None):
            for post in lst:
                if (get_title_from_index(post[0]), post[0]) not in user.nextrecomm and user.visited[post[0]] == 0:
                    user.nextrecomm[i] = (get_title_from_index(post[0]), post[0])
                    break
    
    for i in range(10, 12):
        if (user.nextrecomm[i] == None):
            for post in total_likes:
                if (get_title_from_index(post[0]), post[0]) not in user.nextrecomm and user.visited[post[0]] == 0:
                    user.nextrecomm[i] = (get_title_from_index(post[0]), post[0])
                    break

def main():
    Me = User('5e5dfbbefbc8805f69e02c91')
    Me.findInterests(unique_tags)
    knowledgeBasedRecommendation(Me)
    Me.showRecomm()
    
    while(True):
        
        Me.selectPost()
        if Me.stop == True:
            break
        contentBasedRecommendation(Me)
        knowledgeBasedRecommendation(Me)
        collabrativeRecommendation(Me)
        Me.showRecomm()
            
if __name__ == "__main__":
    main() 
