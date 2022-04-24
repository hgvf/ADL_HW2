import json
import pandas as pd
import argparse
import os
from tqdm import tqdm

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/mnt/nas4/weiwei/ADL/proj2/data")
opt = parser.parse_args()

# open all files
context_path = os.path.join(opt.data_dir, 'context.json')
with open(context_path, 'r') as f:
    context = json.load(f)

stage = ['train', 'valid']
# change dataset to SWAG's format
for s in stage:
    file = s + '.json'
    file_path = os.path.join(opt.data_dir, file)
    with open(file_path, 'r') as f:
        data = json.load(f)

    # open json as dataframe
    df = pd.json_normalize(data)

    df = df.rename(columns={'question': 'sent2', 'relevant': 'label'})

    if s != 'test':
        df = df.drop(['answer.text', 'answer.start'], axis='columns')

    for index, row in tqdm(df.iterrows()):
        if s != 'test':
            df.at[index, 'label'] = row['paragraphs'].index(row['label'])

        df.at[index, 'sent1'] = ' '
        paragraphs = row['paragraphs']
        
        for i in range(4):
            column_name = 'ending' + str(i)
            
            df.at[index, column_name] = context[paragraphs[i]]
        
    if s != 'test':
        df = df.drop(['paragraphs'], axis='columns')

    # save as csv
    if not os.path.exists('./SWAG'):
        os.mkdir('./SWAG')

    save_path = os.path.join('./SWAG', s+'.csv')
    df.to_csv(save_path, encoding='utf-8')

