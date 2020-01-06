import pandas as pd

a_dict = {}
b_dict = {}

with open("../../data/a.txt") as f:
    for line in f:
        line = line.split()
        #print(len(line))
        a_dict[line[1]] = line[0]

with open("../../data/b.txt") as f:
    for line in f:
        line = line.split()
        if len(line) == 1:
            b_dict[line[0]] = 0
        else:
            b_dict[line[0]] = 1

train_data = pd.read_csv('../../data/mirna_tpm.csv')
labels = []

for name in train_data['name']:
    labels.append(b_dict[a_dict[name]])

train_data['label'] = labels
train_data.to_csv('../../data/mirna_tpm_pre.csv')
