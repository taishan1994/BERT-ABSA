import os
import re
import json
import codecs
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


class ProcessGdcqData:
    def __init__(self):
        self.data_path = "./data/gdcq/"
        self.train_file = self.data_path + "ori_data/Train_merge.csv"
        self.data = pd.read_csv(self.train_file, encoding="utf-8")

    def get_ner_data(self):
        res = []
        tmp = {}
        id_set = set()
        for d in self.data.iterrows():
            d = d[1]
            did = d[1]
            aspect = d[2]
            a_start = d[3]
            a_end = d[4]
            opinion = d[5]
            o_start = d[6]
            o_end = d[7]
            category = d[8]
            polary = d[9]
            text = d[10]
            if did not in id_set:
                if tmp:
                    # print(tmp)
                    res.append(tmp)
                id_set.add(did)
                tmp = {}
                tmp['id'] = did
                tmp['text'] = [i for i in text]
                tmp['labels'] = ["O"] * len(text)
            try:
                if aspect != "_":
                    tmp['labels'][int(a_start)] = "B-{}".format(category)
                    for i in range(int(a_start) + 1, int(a_end)):
                        tmp['labels'][i] = "I-{}".format(category)
                if category != "_":
                    tmp['labels'][int(o_start)] = "B-{}".format(polary)
                    for i in range(int(o_start) + 1, int(o_end)):
                        tmp['labels'][i] = "I-{}".format(polary)
            except Exception as e:
                continue

        train_ratio = 0.92
        train_num = int(len(res) * 0.92)
        train_data = res[:train_num]
        dev_data = res[train_num:]

        with open(self.data_path + "ner_data/train.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in train_data]))

        with open(self.data_path + "ner_data/dev.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in dev_data]))

        cates = self.data['Categories'].values.tolist()
        cates = list(set(cates))
        polars = self.data['Polarities'].values.tolist()
        polars = list(set(polars))
        labels = cates + polars
        with open(self.data_path + "ner_data/labels.txt", "w") as fp:
            fp.write("\n".join(labels))

    def get_re_data(self):
        res = []
        tmp = {}
        id_set = set()
        for d in self.data.iterrows():
            d = d[1]
            did = d[1]
            aspect = d[2]
            a_start = d[3]
            a_end = d[4]
            opinion = d[5]
            o_start = d[6]
            o_end = d[7]
            category = d[8]
            polary = d[9]
            text = d[10]

            tmp = {}
            tmp['id'] = did
            tmp['text'] = [i for i in text]
            tmp['start'] = [0] * len(text)
            tmp["end"] = [0] * len(text)
            try:
                if aspect != "_":
                    tmp["aspect"] = aspect
                    if category != "_":
                        tmp['start'][int(o_start)] = 1
                        tmp['end'][int(o_end) - 1] = 1
                    # print(tmp)
                    res.append(tmp)
            except Exception as e:
                continue

        train_ratio = 0.92
        train_num = int(len(res) * 0.92)
        train_data = res[:train_num]
        dev_data = res[train_num:]

        with open(self.data_path + "re_data/train.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in train_data]))

        with open(self.data_path + "re_data/dev.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in dev_data]))


if __name__ == "__main__":
    processGdcqData = ProcessGdcqData()
    processGdcqData.get_ner_data()
    processGdcqData.get_re_data()
