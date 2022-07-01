import json
import os
import numpy as np

def numberFile(file_location):
    count = 0
    for _ in os.listdir(file_location):
        count += 1
    return count

dir = os.getcwd()
count = 0
for filename in os.listdir(dir+r"\豆瓣小组"):
    filepath =  os.path.join(dir+r"\豆瓣小组", filename)
    print(filepath)
    count += 1
    with open(dir+r"\data\merge{}.json".format(count), mode="w+", encoding="utf-8") as fout:
        for i in np.arange(numberFile(filepath)):
            with open(filepath+r"\%d.json"%i, mode="r", encoding="utf-8") as fin: 
                fout.write(fin.read()+"\n")
fin.close()
fout.close()
