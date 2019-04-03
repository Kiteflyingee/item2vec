# coding=utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pickle


def process_data(data_file, sep='\t', test_ratio=0.2):
    '''
    处理数据集，切分训练集和测试集
    '''
    df = pd.read_csv(data_file,
                     sep=sep,
                     header=None,
                     names=['uid', 'iid', 'rating', 'timestamp'])

    # plt.hist(df['rating'], color="#3F5D7D")
    # plt.show()
    df_train, df_test = train_test_split(df,
                                         test_size=test_ratio,
                                         random_state=0,
                                         stratify=df['uid'])
    print("num of train:", len(df_train))
    print("num of test:", len(df_test))
    pickle.dump((df_train, df_test), 
                    file=open('train_test.pkl','wb'))
    return df_train, df_test


def rating_splitter(df):
# def rating_splitter(df, limit=3.5):
    '''
    以 X 为界，标记用户喜欢或者不喜欢 item ,返回item userset
    Params:
        df: DataFrame
        # limit: >limit的item为用户喜欢的item
    Return:
        item:user set
        # a list like [(item_id,userset)]
    '''
    # df['like'] = np.where(df['rating'] > limit, 1, 0)
    # 以喜欢和item id给数据分组
    # group_user_like = df.groupby(['like', 'iid'])
    group_user = df.groupby('iid')
    # 这块需要转换为str,方便word2vec操作
    df['uid'] = df['uid'].astype('str')
    # 把每个item对应的uid返回回去
    return [(gp, group_user.get_group(gp)['uid'].tolist())
            for gp in group_user.groups]

def rating_splitter_item(df):
# def rating_splitter(df, limit=3.5):
    '''
    Params:
        df: DataFrame
    Return:
        item:user set
        # a list like [(user_id, itemset)]
    '''
    # df['like'] = np.where(df['rating'] > limit, 1, 0)
    # 以喜欢和item id给数据分组
    # group_user_like = df.groupby(['like', 'iid'])
    group_item = df.groupby('uid')
    # 这块需要转换为str,方便word2vec操作
    df['iid'] = df['iid'].astype('str')
    # 把每个user的item集合穿回去
    return [(gp, group_item.get_group(gp)['iid'].tolist())
            for gp in group_item.groups]


if __name__ == "__main__":
    data_path = r'./data/ml/u.data'
    df_train, df_test = process_data(data_path)
    # item_users_map = rating_splitter(df_train)
    user_item_map = rating_splitter_item(df_train)

