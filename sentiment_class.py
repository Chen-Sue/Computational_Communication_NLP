# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import utils
from snownlp import SnowNLP
import warnings
warnings.filterwarnings("ignore")

dir = os.getcwd()

topic = 4
for i in np.arange(1, topic+1, 1):
    source = open(dir+r"\data\result_class{}.txt".format(i), 'r', encoding="UTF-8")
    line = source.readlines()

    sentimentslist, result = [], [] 
    for key, value in enumerate(line):
        s = SnowNLP(value)
        sentimentslist.append(s.sentiments)  # 计算情感得分
        result.append(s.sentiments-0.5)

    fig = plt.figure(figsize=(12, 5)) 
    ax = fig.add_subplot(111) 
    plt.hist(sentimentslist, bins=np.arange(0,1,0.01), facecolor='dodgerblue')
    plt.xlabel('Sentiments Probability', fontsize=20)
    plt.ylabel('Number of Comments', fontsize=20)
    plt.title('Analysis of Sentiments (topic={})'.format(i), fontsize=24)
    plt.tick_params(axis='both', labelsize=16)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
    plt.tight_layout()
    plt.savefig(r".\figure\sentiment{}.png".format(i))
    # plt.show()

    fig = plt.figure(figsize=(14, 10)) 
    ax = fig.add_subplot(211) 
    plt.scatter(list(np.arange(0, len(sentimentslist), 1)), sentimentslist)
    plt.xlabel('Index of Comments', fontsize=20)
    plt.ylabel('Sentiment Probability', fontsize=20)
    plt.title('Analysis of Sentiments (topic={})'.format(i), fontsize=24)
    plt.tick_params(axis='both', labelsize=16)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax = fig.add_subplot(212) 
    plt.axhline(y=0, ls="-", c="red")
    plt.scatter(list(np.arange(0, len(sentimentslist), 1)), result)
    plt.xlabel('Index of Comments', fontsize=20)
    plt.ylabel('Sentiment Probability', fontsize=20)
    plt.title('Analysis of Sentiments (topic={})'.format(i), fontsize=24)
    plt.tick_params(axis='both', labelsize=16)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(r".\figure\number{}.png".format(i))
    # plt.show()