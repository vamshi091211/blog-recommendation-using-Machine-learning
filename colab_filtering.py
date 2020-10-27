#COLLABORATIVE FILTERING

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
def get_similar_posts(post_category):
  similar_score=item_similarity_df[post_category]
  similar_score=similar_score.sort_values(ascending=False)
  return similar_score
photo_lover=['Faith in yourself','Keep working hard !!']
def get_category_from_title(title):
  return merged_df[merged_df['title']==title]['new_category'].values[0]
similar_post=pd.DataFrame()
for post_name in photo_lover:
  similar_post=similar_post.append(get_similar_posts(get_category_from_title(post_name)),ignore_index=True)
df1=similar_post.sum().sort_values(ascending=False)
pd.set_option('display.max_rows', df1.shape[0]+1)
print(df1)
df2=pd.DataFrame(df1,columns=['Score'])
df2.columns
df2.head()
modified=df2.reset_index()
modified.columns
modified.rename(columns={'index':'Category'},inplace=True)
modified.columns
modified.head(3)
res=[]
for x in modified['Category']:
  res.append(merged_df.loc[merged_df['new_category']==x,['title']].values[0])
modified['Title']=np.array(res)
modified.head(3)
final_modified=modified[['Title','Category','Score']].drop(['Score'],axis=1)
final_modified