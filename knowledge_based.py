for i in range(len(post_info['category'])):
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

post_info[' post_type'].value_counts()
unique_tags.pop(0)

k = 0
for i in range(79):
    for j in range(3):
        if(k < len(unique_tags)):
            print(unique_tags[k].center(40), end = "|")
        k+=1
    print("\n")
    
print("choose tags as per your interest")
interests = []
while(True):
    i = input()
    if(len(i) == 0):
        break
    interests.append(i)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_kernels

target = " ".join(interests)
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
    if(i > 10):
        break
    print(get_title_from_index(second))
    i+=1
