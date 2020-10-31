class User:
    
    def __init__(self, user_id):
        self.userid = user_id
        self.interests = []
        self.visited = [0]*500
        self.nextrecomm = []
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

        for name, index in self.nextrecomm:
            print(index, name)
            
    def selectPost(self):
        recomindex = []
        for x in self.nextrecomm:
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
            if (self.nextrecomm[i][1] == n):
                garbage = self.nextrecomm.pop(i)
                break
        print("\n")
        
    def resetrecomm(self):
        self.nextrecomm = []
        
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
post_info['new_category'] = [x.split('|')[0] for x in post_info['category']]
post_info['new_category'] = [x.split(';')[0] for x in post_info['new_category']]
post_info['new_category'].replace({'ViDEO':'Video'},inplace=True)
df = pd.DataFrame(post_info['new_category'])
new_post_info=post_info.rename(columns = {'_id':'post_id'})
merged_df=pd.merge(new_post_info,user_and_post_info).drop(['timestamp','combined_features','index'],axis=1)
merged_df['rating']=1
user_ratings=merged_df.pivot_table(index=['user_id'],columns=['new_category'],values='rating')
user_ratings=user_ratings.fillna(0)
item_similarity_df = user_ratings.corr(method='pearson')
def get_similar_posts(post_category):
    similar_score =item_similarity_df[post_category]
    similar_score=similar_score.sort_values(ascending=False)
    return similar_score

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
    if len(user.nextrecomm) > 2:
        _ = user.nextrecomm.pop(0)
        _ = user.nextrecomm.pop(0)
    target = " ".join(user.interests)
    candidates = list(post_info["category"])
    vec = CountVectorizer()
    vec.fit(candidates)
    cos_sim = pairwise_kernels(vec.transform([target]), vec.transform(candidates), metric = "cosine")
    similarity_pair = []
    for i in range(len(cos_sim[0])):
        similarity_pair.append((cos_sim[0][i], i))
    similarity_pair.sort(reverse = True)
    i = 0
    for first, second in similarity_pair:
        if(len(user.nextrecomm) == 4 or len(user.nextrecomm) >= 10):
            break
        if (get_title_from_index(second), second) not in user.nextrecomm and user.visited[second] == 0:
            user.nextrecomm.append((get_title_from_index(second), second))
        i+=1
        
        
def contentBasedRecommendation(user):
    if len(user.nextrecomm) > 2:
        _ = user.nextrecomm.pop(0)
        _ = user.nextrecomm.pop(0)  
    post_index = user.lastpost
    # Passing the post_index to cosine_sim to make a list of similar posts
    similar_posts = list(enumerate(cosine_sim[post_index]))
    # Sorting the list in decreasing order of their similarity
    sorted_similar_posts = sorted(similar_posts, key = lambda x:x[1], reverse=True)
    temp = sorted_similar_posts.pop(0)
    i = 0
    for post in sorted_similar_posts:
        if len(user.nextrecomm) == 7 or len(user.nextrecomm) >= 10:
            break
        if (get_title_from_index(post[0]), post[0]) not in user.nextrecomm and user.visited[post[0]] == 0:
            user.nextrecomm.append((get_title_from_index(post[0]), post[0]))
        i = i + 1
        
            
def collabrativeRecommendation(user):
    photo_lover=['Faith in yourself','Keep working hard !!']
    similar_post=pd.DataFrame()
    for post_name in photo_lover:
        similar_post=similar_post.append(get_similar_posts(get_category_from_title(post_name)),ignore_index=True)
    df1=similar_post.sum().sort_values(ascending=False)
    pd.set_option('display.max_rows', df1.shape[0]+1)
    print(df1)
    df2=pd.DataFrame(df1,columns=['Score'])
    modified=df2.reset_index()
    modified.rename(columns={'index':'Category'},inplace=True)
    res=[]
    for x in modified['Category']:
        res.append(merged_df.loc[merged_df['new_category']==x,['title']].values[0])
    modified['Title']=np.array(res)
    final_modified=modified[['Title','Category','Score']].drop(['Score'],axis=1)
    
def main():
    Me = User('5e5dfbbefbc8805f69e02c91')
    Me.findInterests(unique_tags)
    knowledgeBasedRecommendation(Me)
    Me.showRecomm()
    
    while(True):
        
        Me.selectPost()
        if Me.stop == True:
            break
        Me.resetrecomm()
        contentBasedRecommendation(Me)
        knowledgeBasedRecommendation(Me)
        Me.showRecomm()
            
if __name__ == "__main__":
    main()    
