import json
import os
import numpy as np

def numberFile(file_location):
    count = 0
    for _ in os.listdir(file_location):
        count += 1
    return count

# 合并数据
dir = os.getcwd()
with open(dir+r"\data\merge.json",mode="w+",encoding="utf-8") as fout:
    for filename in os.listdir(dir+r"\豆瓣小组"):
        filepath =  os.path.join(dir+r"\豆瓣小组", filename)
        for i in np.arange(numberFile(filepath)):
            with open(filepath+r"\%d.json"%i,mode="r",encoding="utf-8") as fin: 
                fout.write(fin.read()+"\n")
fin.close()
fout.close()

# with open(dir+r"\data\merge.json", mode="w+", encoding="utf-8") as fin1:
#     for filename in os.listdir(dir+r"\豆瓣小组"):
#         filepath =  os.path.join(dir+r"\豆瓣小组", filename)
#         for i in np.arange(numberFile(filepath)):
#             with open(filepath+r"\%d.json"%i, mode="r", encoding="utf-8") as fin: 
#                 fin1.write(fin.read()+"\n")
#                 for line in fin1.readlines():
#                     #替换json中的\n字符为空字符
#                     line = line.replace("\\n","")
#                     print(line)
#                     #将json格式转化为python格式
#                     lines = json.loads(line) 
#                     #提取json数据中的title和content的内容
#                     ti = lines["title"]
#                     content = lines["content"]
#                     with open(dir+r"\data\merge.txt", mode="w+", encoding="utf-8") as fout:
#                         fout.write(ti)
#                         fout.write(content)
# fin.close()
# fout.close()