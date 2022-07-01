
import json
import os

dir = os.getcwd()

# 提取json数据中的comments的内容
with open(dir+r"\data\comments.txt", mode="w+", encoding="utf-8") as fout:
    with open(dir+r"\data\merge.json", mode="r", encoding="utf-8") as fin:
        for line in fin.readlines():
            line = line.replace("fire", "FIRE")
            try: lines = json.loads(line) #将json格式转化为python格式
            except Exception as e: continue
            comments = lines["comments"]
            print(comments)
            try: fout.write(str(comments))
            except Exception as e: continue
fin.close()
fout.close()
