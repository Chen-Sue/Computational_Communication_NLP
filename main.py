# -*- coding:utf-8 -*-
import jieba
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import os
import PIL.Image as image
from wordcloud import WordCloud
import pandas as pd
import matplotlib
import jieba.analyse
import pyLDAvis
import pyLDAvis.sklearn
import joblib
import warnings
warnings.filterwarnings("ignore")

import utils

dir = os.getcwd()
font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\STXINWEI.TTF')

stopword_list = utils.get_stopword_list(dir+r"\data\stopwords.txt")
comments = open(dir+r"\data\comments.txt", 'r', encoding="UTF-8")
# 分词并去除停用词
corpus = []
for doc in comments: 
    comments_cut = [temp for temp in jieba.cut(doc) \
        if len(temp)>1 and temp not in stopword_list]
    corpus.append(' '.join(comments_cut))

# 使用Tfidf或Count方法初始化
# vectorizer = TfidfVectorizer(max_features=4000, use_idf=True, smooth_idf=True, 
#     stop_words=stopword_list, , norm=Nonebinary=False, decode_error='ignore') 
vectorizer = CountVectorizer(max_features=4000)
sklearn_dtm = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names() 
print('词表大小:', len(vectorizer.vocabulary_))
print(sklearn_dtm.shape)

wordcloud = WordCloud(font_path='C:\Windows\Fonts\STXINWEI.TTF', margin=5, 
    width=1500, height=960, background_color="white", 
    mask=np.array(image.open(dir+r'\figure\background.jpg')), 
    max_words=800, max_font_size=400, random_state=42)

perplexity_list = []
sep, max_topic = 1, 4
topic_list = np.arange(2, max_topic+1, sep)

with open(dir+r"\data\result_topic_list.txt", 'w') as fw:
    for key,num_topics in enumerate(topic_list):
        lda = LatentDirichletAllocation(n_components=num_topics, \
            max_iter=100, n_jobs=-1, random_state=100) # LDA
        docres = lda.fit_transform(sklearn_dtm)
        perplexity_values = lda.perplexity(sklearn_dtm) #计算困惑度
        fw.write("{}, {:.2f}\n".format(num_topics, perplexity_values))
        joblib.dump(lda, dir+r'\model\lda{}.pkl'.format(num_topics))
        print(f"num. topic: {num_topics}, Perplexity: {perplexity_values}")
        perplexity_list.append(perplexity_values)
        vis = pyLDAvis.sklearn.prepare(lda, sklearn_dtm, vectorizer) # 画分类图
        pyLDAvis.save_html(vis, dir+r"\figure\lda_pass{:.0f}.html".format(num_topics))
        # pyLDAvis.show(vis,open_browser=True) # http://127.0.0.1:8888/

perplexity_list = np.loadtxt(dir+r"\data\result_topic_list.txt",\
    delimiter=',', usecols=(1))
# ind = np.argmin(np.array(perplexity_list))
# topic = perplexity_list[ind]
topic = 4
lda = joblib.load(dir+r'\model\lda{}.pkl'.format(topic))
docres = lda.fit_transform(sklearn_dtm)

LDA_corpus = np.array(docres)
print('类别所属概率:\n', LDA_corpus)
print('主题词所属矩阵：\n', lda.components_)

LDA_corpus_one = np.zeros([LDA_corpus.shape[0]])
LDA_corpus_one = np.argmax(LDA_corpus, axis=1) 
print('每个文档所属类别：', LDA_corpus_one, len(LDA_corpus_one))

with open(dir+r"\data\comments.txt", 'r', encoding="UTF-8") as fin:
    comments = fin.readlines()

for i in np.arange(topic): # 将不同类别的楼主分别合并到不同的文件中
    with open(dir+r"\data\result_class{}.txt".format(i+1), \
        mode="w+", encoding="utf-8") as fin:
        for key1, j in enumerate(LDA_corpus_one):
            if i == j:
                for key, value in enumerate(comments):
                    if key1 == key:
                        fin.write(value)

# 将概率归一化到0到1之间
tt_matrix = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
with open(dir+r"\data\result_topic_weight{}.txt".format(topic), 'w') as fw:
    for id,tt_m in enumerate(tt_matrix):
        tt_m = np.around(tt_m, 4)
        tt_dict = [(name, tt) for name, tt in zip(feature_names, tt_m)]
        tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
        tt_dict = tt_dict[:10]
        print('主题%d:' % (id+1), tt_dict)
        fw.write("Topic #{}\n".format(id+1))
        fw.write("{}\n".format(tt_dict))
        
utils.graph_draw(topic_list, perplexity_list, sep)
utils.print_top_words(lda, feature_names, topic, n_top_words=10)
utils.plot_top_words(lda, feature_names, topic, n_top_words=10)
utils.plot_wordcloud(lda, feature_names, wordcloud, topic, n_top_words=100)

