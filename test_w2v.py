import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from process_data import *

# word_vectors = KeyedVectors.load('item2vec', mmap='r')
# word_vectors.wv

# train user vector

df_train, df_test = pickle.load(open('train_test.pkl','rb'))
item_userlist = rating_splitter(df_train)
