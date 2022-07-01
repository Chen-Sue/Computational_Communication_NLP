import os
import PIL.Image as image
import numpy as np
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud, STOPWORDS

import utils

dir = os.getcwd()
data_location = os.path.join(dir, "figure")
if not os.path.exists(data_location):
    os.mkdir(data_location)

stopword_list = utils.get_stopword_list(dir+r"\data\stopwords.txt")

with open(dir+r"\data\comments.txt", 'r', encoding="UTF-8") as file:
    content = "".join(file.readlines())
word_list = jieba.cut(content) # 精确模式
cut_word = " ".join(word_list)

content_after = ''
for word in cut_word:
    if word not in stopword_list:
        content_after += word + " "

stopwords = set(STOPWORDS)
# stopwords.add("said")

mask=np.array(image.open(dir+r'\figure\background.jpg')) 

wc = WordCloud(width=1500, height=960, font_path="simhei.ttf", 
    # max_words=100, max_font_size=150, random_state=50
    stopwords=stopwords, mask=mask, background_color="white")
wc.generate(content_after)
wc.to_file(dir+r'\figure\cleanloud.png')    # 保存到本地图片文件

plt.figure()
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()