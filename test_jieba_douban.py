import os
import jieba
from collections import Counter

import utils
dir = os.getcwd()

word_list = []
stopword_list = utils.get_stopword_list(dir+r"\data\stopwords.txt")

with open(dir+r"\data\comments.txt", 'r', encoding="UTF-8") as fin:
    comments = "".join(fin.readlines())
words = [word for word in jieba.cut(comments, cut_all=False) \
    if len(word)>=2 and word not in stopword_list]
for text in words:
	word_list.append([text])
word_str = " ".join(words)

c = Counter(words)
for word_freq in c.most_common(200):
	word, freq = word_freq
	print(word, freq)

print(word_str)
with open(dir+r"\data\jieba.txt", mode="w+", encoding="utf-8") as fout:
	fout.write(word_str)

fout.close()
fin.close()