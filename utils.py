import matplotlib.pyplot as plt
from gensim import corpora, models
import math
from gensim.models import LdaModel
import numpy as np
import pandas as pd
import jieba
import os
import matplotlib
font = matplotlib.font_manager.FontProperties(
    fname='C:\Windows\Fonts\STXINWEI.TTF')
dir = os.getcwd()

def get_stopword_list(file): # 读取停用词列表
    with open(file, 'r', encoding='utf-8') as f:  
        # stopword_list = [word.strip('\n') for word in f.readlines()]
        text = f.read()
        stopword_list = text.split('\n')
    return stopword_list

def clean_stopword(word_list, stopword_list): # 清除停用词语
    result = []
    for w in word_list:
        if w not in stopword_list:
            result.append(w)
    return result

def graph_draw(topic, perplexity, sep): 
    fig = plt.figure(figsize=(12, 5)) 
    ax = fig.add_subplot(111) 
    plt.plot(topic, perplexity, c="dodgerblue", markersize=8, linewidth=2, marker='o')
    plt.xlabel("Num. Topics", fontsize=16)
    plt.ylabel("Perplexity", fontsize=16)
    # for a,b in zip(topic, perplexity):
    #     plt.text(a, b+0.1, '%.2f'%b, ha='center', va='bottom', fontsize=12)
    plt.tick_params(axis='both',labelsize=14)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(sep))
    plt.tight_layout()
    plt.savefig(r".\figure\topic-perplexity.png")
    plt.show()
 
def ldamodel(num_topics, word_list):
    dictionary = corpora.Dictionary(word_list)
    corpus = [dictionary.doc2bow(text) for text in word_list]  
    corpora.MmCorpus.serialize('corpus.mm', corpus)
    lda = LdaModel(corpus=corpus, id2word=dictionary, 
        random_state=100, iterations=10, num_topics=num_topics)
    topic_list = lda.print_topics(num_topics, num_words=10)
    for topic in topic_list:
        print(topic)
    return lda, dictionary

def perplexity(ldamodel, testset, dictionary, num_topics):
    print('the info. ldamodel:')
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] 
    size_dictionary = len(dictionary.keys())
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)  
    doc_topics_ist = []  
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0  # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0  # the num of words in the doc
        for word_id, num in dict(doc).items():
            prob_word = 0.0  # the probablity of the word
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic * prob_topic_word
            prob_doc += math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  
    # perplexity = exp(-sum(p(d)/sum(Nd))
    print("the perplexity of this ldamodel is {:.2f}".format(prep), prob_doc_sum / testset_word_num)
    return prep

def print_top_words(model, feature_names, topic, n_top_words):
    # with open(dir+r"\data\result_topic_number{}.txt".format(topic), 'w') as fw:
        for topic_idx,topic in enumerate(model.components_):
            # fw.write("Topic #{}\n".format(topic_idx+1))
            # fw.write("(word:{}, componets: {}) \n".format(
            #     [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]],
            #     topic[topic.argsort()[:-n_top_words-1:-1]]))
            print("Topic {}".format(topic_idx+1))
            print(" ".join([feature_names[i] 
                for i in topic.argsort()[:-n_top_words-1:-1]]))

def plot_top_words(model, tf_feature_names, topic, n_top_words):
    fig = plt.figure(figsize=(16,5))
    components = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    for topic_id, topic in enumerate(components):
        topword = pd.DataFrame({"word":[tf_feature_names[i] \
            for i in topic.argsort()[:-n_top_words-1:-1]],
            "componets":topic[topic.argsort()[:-n_top_words-1:-1]]})
        topword.sort_values(by="componets").plot(
            kind="barh", x="word", y="componets",\
            legend=False, color='cornflowerblue', subplots=True, 
            ax=fig.add_subplot(1,4,topic_id+1))
        plt.yticks(FontProperties=font)
        plt.ylabel("")
        plt.xlabel("probability", FontProperties=font, size=20)
        plt.tick_params(axis='both', labelsize=16)
        plt.title("Topic %d" %(topic_id+1), FontProperties=font, size=24)
    plt.tight_layout(w_pad=2)
    plt.savefig(dir+r'\figure\top_words.png')
    plt.show()

def plot_wordcloud(model, tf_feature_names, wordcloud, topic, n_top_words):
    _ = plt.figure(figsize=(16,4))
    for topic_id, topic in enumerate(model.components_):
        topword = pd.DataFrame({"word":[tf_feature_names[i] \
            for i in topic.argsort()[:-n_top_words-1:-1]],
            "componets":topic[topic.argsort()[:-n_top_words-1:-1]]})
        topword = topword.sort_values(by="componets")
        word_dict = {}
        for key,value in zip(topword.word,topword.componets):
            word_dict[key] = round(value)
        plt.subplot(1, 4, topic_id+1)
        plt.imshow(wordcloud.generate_from_frequencies(frequencies=word_dict))
        plt.axis('off')
        plt.title("Topic %d" %(topic_id+1), FontProperties=font, size=24)
    plt.tight_layout(w_pad=3)
    plt.savefig(dir+r'\figure\wordcloud.png')
    plt.show()

def clean_text(text, stopwords):
    wordlist = jieba.lcut(text) 
    wordlist = [w for w in wordlist if w not in stopwords and len(w)>1]
    document =  " ".join(wordlist)
    return document

def top_words_data_frame(model, tf_idf_vectorizer, n_top_words):
    '''
    求出每个主题的前 n_top_words 个词

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation 
    tf_idf_vectorizer : sklearn 的 TfidfVectorizer
    n_top_words :前 n_top_words 个主题词

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    rows = []
    feature_names = tf_idf_vectorizer.get_feature_names()
    for topic in model.components_:
        top_words = [feature_names[i]
                     for i in topic.argsort()[:-n_top_words - 1:-1]]
        rows.append(top_words)
    columns = [f'topic {i+1}' for i in range(n_top_words)]
    df = pd.DataFrame(rows, columns=columns)

    return df


def predict_to_data_frame(model, X):
    '''
    求出文档主题概率分布情况

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation 
    X : 词向量矩阵

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    matrix = model.transform(X)
    columns = [f'P(topic {i+1})' for i in range(len(model.components_))]
    df = pd.DataFrame(matrix, columns=columns)
    return df
