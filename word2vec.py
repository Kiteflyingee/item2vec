#coding=utf-8

import gensim
import random
from gensim.models import Word2Vec
import datetime
import process_data
# 引入日志配置
import logging


def shuffle_data(item_users):

    for userlist in item_users:
        random.shuffle(userlist)
    
def word_to_vec(user_sequence, model_save_path=r'item2vec'):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    assert gensim.models.word2vec.FAST_VERSION > -1
    start = datetime.datetime.now()
    print("start word2vec train...")
    model_w2v_sg = Word2Vec(sentences = user_sequence,
                        iter = 50000, # epoch
                        min_count = 3, # a user has to appear more than 3 times to be keeped
                        size = 300, # size of the hidden layer
                        workers = 12, # specify the number of threads to be used for training
                        sg = 1,
                        hs = 0,
                        negative = 5,
                        window = 9999999)
    print("Time passed: " + str(datetime.datetime.now()-start))
    model_w2v_sg.save(model_save_path)
    return model_w2v_sg

def userCF_recommender(model, train, userid, k=10, topN=10):
    '''
    usercf推荐，word2vec做user2vec时候用

    返回: a list , length=topN
            [(iid, score)]
    '''
    # 这里考虑不用gensim，因为每个词向量需要重新用auto-encoder
    most_similar_users = model.wv.most_similar_cosmul(positive=[userid], topn=k)
    witch_item = train[userid]
    item_scores = {}
    for uid, sim in most_similar_users:
        items = train[uid]
        for iid in items:
            if iid in witch_item:
                continue
            item_scores.setdefault(iid, 0)
            item_scores[iid] += sim
    return sorted(item_scores.items, lambda x:x[1], reverse=True)[0:topN]


# make itemCF Recommender，当训练的是item2vec才有用
def itemCF_recommender(model, positive_list=None, negative_list=None,topN=10):
    '''
    用作itemcf时候，当且仅当word2vec三训练的item2vec
    '''
    recommend_list = []

    most_similar_list = model.wv.most_similar_cosmul(positive=positive_list, negative=negative_list, topn=topN)
    for iid, prob in most_similar_list:
        recommend_list.append(iid)
        logging.log(iid, prob)
    return recommend_list


def scores_at_m (model, test_buy, recommend_buy, topn=10):
    '''
    :test_buy   dict 
                userid:itemset
    :recommend_buy  dict
                userid:itemset
    '''
    sum_liked = 0
    sum_correct = 0
    common_users = set(test_buy.keys()).intersection(set(recommend_buy.keys()))

    for userid in common_users:
        current_test_set = set(test_buy[userid])
        pred = [pred_result[0] for pred_result in model.wv.most_similar_cosmul(positive = recommend_buy[userid], topn=topn)]
        sum_correct += len(set(pred).intersection(current_test_set))
        sum_liked += len(current_test_set)
    precision_at_m = sum_correct/(topn*len(common_users))
    recall_at_m = sum_correct/sum_liked
    f1 = 2/((1/precision_at_m)+(1/recall_at_m))
    return [precision_at_m, recall_at_m, f1]

def train_user_vector():
    data_path = r'./data/ml/u.data'
    df_train, df_test = process_data.process_data(data_path)
    item_users_map = process_data.rating_splitter(df_train)
    itemids, user_sequence = zip(*item_users_map)
    user_sequence = list(user_sequence)
    shuffle_data(user_sequence)

    start = datetime.datetime.now()
    print("start word2vec train...")
    
    model_w2v_sg = Word2Vec(sentences = user_sequence,
                        iter = 50000, # epoch
                        min_count = 3, # a user has to appear more than 3 times to be keeped
                        size = 300, # size of the hidden layer
                        workers = 12, # specify the number of threads to be used for training
                        sg = 1,
                        hs = 0,
                        negative = 5,
                        window = 9999999)
    print("Time passed: " + str(datetime.datetime.now()-start))
    model_w2v_sg.save('user2vec')


def train_item_vector():
    data_path = r'./data/ml/u.data'
    df_train, df_test = process_data.process_data(data_path)
    user_items_map = process_data.rating_splitter_item(df_train)
    userids, item_sequence = zip(*user_items_map)
    item_sequence = list(item_sequence)
    shuffle_data(item_sequence)

    start = datetime.datetime.now()
    print("start word2vec train...")
    
    model_w2v_sg = Word2Vec(sentences = item_sequence,
                        iter = 50000, # epoch
                        min_count = 3, # a user has to appear more than 3 times to be keeped
                        size = 300, # size of the hidden layer
                        workers = 12, # specify the number of threads to be used for training
                        sg = 1,
                        hs = 0,
                        negative = 5,
                        window = 9999999)
    print("Time passed: " + str(datetime.datetime.now()-start))
    model_w2v_sg.save('item2vec')

if __name__ == "__main__":

    train_item_vector()