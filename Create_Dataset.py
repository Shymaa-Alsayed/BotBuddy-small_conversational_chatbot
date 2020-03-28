import yaml
import os
import pickle

questions = []
categories = []
CQA = {}


files_path = 'chatterbotenglish/'
files_names = [f for f in os.listdir(files_path)]
for name in files_names:
    with open(files_path+name, 'r') as file:
        data = yaml.safe_load(file)
    category=data['categories'][0]
    CQA[category]={}
    for pair in data['conversations']:
        temp_q=pair[0]
        temp_a=pair[1:]
        if temp_q not in CQA[category]:
            questions.append(temp_q)
            categories.append(category)
            CQA[category][temp_q]=[]
            for a in temp_a:
                CQA[category][temp_q].append(a)
        else:
            CQA[category][temp_q].extend(temp_a)

pickle.dump({'questions':questions, 'labels':categories, 'CQA':CQA},open("data", "wb"))