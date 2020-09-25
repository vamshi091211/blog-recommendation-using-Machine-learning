import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity  
user_info=pd.read_csv("user.csv",encoding='ISO-8859-1')
user_and_post_info=pd.read_csv("view.csv")
post_info=pd.read_csv("post.csv")
user_info.shape
user_info.head()
user_info.describe()
user_info.columns
df=user_info.loc[(user_info['gender']!='male')& (user_info['gender']!='female')]
df
user_and_post_info.describe()
user_and_post_info.columns
post_info.columns
user_info.columns
post_info.describe()
user_ids=pd.DataFrame(user_info['_id'])
#post_ids=pd.DataFrame(post_info['_id'])
post_info.insert(0, 'index', range(0, len(post_info)))
post_info.head()
post_info.dtypes
user_ids
post_info['_id']=post_info['_id'].astype(str)
post_info.dtypes
features=['title','category',' post_type']
for feature in features:
  post_info[feature]=post_info[feature].fillna("")
def combine_features(row):
  try:
    return row['title']+" "+row['category']+" "+row[' post_type']
  except:
    print ("Error:",row)


post_info["combined_features"]=post_info.apply(combine_features,axis=1)
post_info["combined_features"].head()
cv=CountVectorizer()
count_matrix=cv.fit_transform(post_info["combined_features"])
cosine_sim=cosine_similarity(count_matrix)
def get_index_from_title(title):
  return post_info[post_info['title']==title]['index'].values[0]
def get_title_from_index(index):
  return post_info[post_info['index']==index]['title'].values[0]
def get_post_id_from_user_id(userid):
  return user_and_post_info[user_and_post_info['user_id']==userid]['post_id'].values[0]
def get_title_from_post_id(postid):
  return post_info[post_info['_id']==postid]['title'].values[0]
#post_user_likes='MIS'
Id_of_user='5e5dfbbefbc8805f69e02c91'
Id_of_post_user_likes=get_post_id_from_user_id(Id_of_user)
post_user_likes=get_title_from_post_id(Id_of_post_user_likes)
print(post_user_likes)
post_index=get_index_from_title(post_user_likes)
similar_posts=list(enumerate(cosine_sim[post_index]))
sorted_similar_posts=sorted(similar_posts,key=lambda x:x[1],reverse=True)
i=0
for post in sorted_similar_posts:
  print(get_title_from_index(post[0]))
  i=i+1
  if i>50:
    break
post_info['category'].unique
post_info['new_category'] = [x.split('|')[0] for x in post_info['category']]
post_info['new_category'] = [x.split(';')[0] for x in post_info['new_category']]
post_info = post_info[post_info.new_category != '']
post_info['new_category'].replace({'ViDEO':'Video'},inplace=True)
post_info['new_category'].nunique()
df=pd.DataFrame(post_info['new_category'])
df
post_info=post_info.drop(['category'],axis=1)
post_info.columns
new_post_info=post_info.rename(columns = {'_id':'post_id'})
new_post_info.head()
merged_df=pd.merge(new_post_info,user_and_post_info).drop(['timestamp','combined_features','index'],axis=1)
merged_df.shape
merged_df['rating']=1
merged_df
user_ratings=merged_df.pivot_table(index=['user_id'],columns=['new_category'],values='rating')
user_ratings.head()
user_ratings=user_ratings.fillna(0)
item_similarity_df=user_ratings.corr(method='pearson')
item_similarity_df.head(3)

