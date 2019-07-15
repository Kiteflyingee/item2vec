# 组织结构说明  

## process_data.py和process.py  
数据集预处理文件，分别针对训练item的词表示和user的词表示  

## word2vec.py
实现了item2vec与user2vec，user2vec纯属自己瞎想，实验效果不理想

## vea.py
用keras实现的简单的变分自动编码器代码，想通过autoencoder获得新的item向量表示，对item的向量表示做个压缩，试试看词表示的效果会不会有惊喜


# 实验过程记录
1. 数据集已经切分好了
2. user2vec在新数据集下已经失效，需要重新训练


# TO DO 
等待word2vec训练完成。。。  
1. 查看word2vec相似度矩阵形式  
2. 根据相似性矩阵形式融合UserCF  
3. 对比item2vec模型itemCF  
4. 使用keras搭建AE或者VAE  

