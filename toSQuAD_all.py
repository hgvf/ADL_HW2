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

# change dataset to SQuAD's format
for s in stage:
    file_path = os.path.join(opt.data_dir, s+'.json')
    with open(file_path, 'r') as f:
        data = json.load(f)

    # open json as dataframe
    df = pd.json_normalize(data)

    df = df.drop(['paragraphs'], axis='columns')
    
    tmp = []
    for index, row in tqdm(df.iterrows()):
        df.at[index, 'context'] = context[row['relevant']]
        
    df = df.drop(['relevant'], axis='columns')
   
    # save as csv
    if not os.path.exists('./SQuAD'):
        os.mkdir('./SQuAD')

    save_path = os.path.join('./SQuAD', s+'.csv')
    df.to_csv(save_path, encoding='utf-8')
