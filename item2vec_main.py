
import gensim
from gensim.models import Word2Vec
from word2vec import *
from process_data import *
import pickle
# itemCF

TOPN = 10
model = Word2Vec.load(r"./item2vec")
df_train, df_test = pickle.load(open('train_test.pkl','rb'))
user_itemslist = rating_splitter_item(df_train)
users , itemlist = zip(*user_itemslist)
itemslist = list(itemlist)
del itemlist
# users 表示user的数组,索引和itemslist一样，元素为user对应userid,itemslist对应元素为user购买的item集合

for idx in range(len(users)):
    user = users[idx]
    items = itemslist[idx]
    
    item_score = {}# 存储item可能的推荐列表
    for item in items:
        # 获得与该item相似的k个item
        if item in model.wv.index2word:
            items_sim = itemCF_recommender(model, item, k=10)
            for i_sim in items_sim:
                item_score.setdefault(i_sim[0], 0)
                item_score[i_sim[0]] += i_sim[1]
    recommendlist = sorted(item_score.items(), key=lambda x:x[1], reverse=False)[0:TOPN]
    print(recommendlist)
    