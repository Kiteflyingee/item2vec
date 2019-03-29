import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

word_vectors = KeyedVectors.load('item2vec', mmap='r')
word_vectors.wv