
import gensim
from gensim.models import Word2Vec
from word2vec import *
import process_data

# itemCF

data_path = r'./data/ml/u.data'
model = Word2Vec.load(r"./item2vec")
df_train, df_test = process_data.process_data(data_path)
groupitem = df_train.groupby('iid')
train = {}
for gp in groupitem:
    train[gp] = groupitem.get_group(gp)['uid']

user_items = process_data.rating_splitter_item(df_test)
users, itemlist = zip(*user_items)
itemtolist = list(itemlist)
del itemlist
for idx in range(len(users)):
    userid = str(users[idx])
    buy_item = itemtolist[idx]
    # userCF_recommender(model, train,positive_list=[userid])