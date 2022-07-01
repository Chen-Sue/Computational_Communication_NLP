from gensim import corpora, models
from gensim.models import LdaModel
import numpy as np
import jieba
import os
from collections import defaultdict
from gensim.corpora.dictionary import Dictionary

import utils

if __name__ == '__main__':
    print("perplexity")
    dir = os.getcwd()
    topic_list = np.arange(1, 10, 1)
    perplexity_list = []
    word_list = []

    stopword_list = utils.get_stopword_list(dir+r"\data\stopwords.txt")
    with open(dir+r"\data\comments.txt", 'r', encoding="UTF-8") as file:
        texts = "".join(file.readlines())
    texts = [word for word in jieba.cut(texts, cut_all=False) \
        if len(word)>=2 and word not in stopword_list]
    
    # 去掉只出现一次的单词
    frequency = defaultdict(int)
    for text in texts:
        frequency[text] += 1
    for text in texts:
        if frequency[text]>1:
            word_list.append([text])

    dictionary = Dictionary(word_list)
    corpus = [dictionary.doc2bow(words) for words in word_list]

    for num_topics in topic_list:
        print('\nnum of topics: %s' % num_topics)
        
        # lda, dictionary = utils.ldamodel(num_topics, word_list)
        # corpus = corpora.MmCorpus('corpus.mm')
        lda = models.ldamodel.LdaModel(corpus=corpus[:int(len(corpus)*0.8)], id2word=dictionary, 
            # update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True
            num_topics=num_topics, random_state=100, iterations=10)
        
        perplexity_values = utils.perplexity(lda, corpus[int(len(corpus)*0.8):], 
            dictionary, num_topics)
        print('%d 个主题的Perplexity为:' % (num_topics), perplexity_values)
        # perwordbound = lda.log_perplexity(corpus[int(len(corpus)*0.8):])
        # perplexity_values = np.exp2(-perwordbound)
        # print('%d 个主题的Perplexity为:' % (num_topics), perwordbound, perplexity_values)
        perplexity_list.append(perplexity_values)

    utils.graph_draw(topic_list, perplexity_list)
