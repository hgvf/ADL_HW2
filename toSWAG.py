import json
import argparse
import pandas as pd
from tqdm import tqdm

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--context_path", type=str, default="/mnt/nas4/weiwei/ADL/proj2/data/context.json")
parser.add_argument("--test_path", type=str, default='/mnt/nas4/weiwei/ADL/proj2/data/test.json')
opt = parser.parse_args()

# open all files
with open(opt.context_path, 'r') as f:
    context = json.load(f)

s = 'test'

with open(opt.test_path, 'r') as f:
    data = json.load(f)

# change dataset to SWAG's format
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
df.to_csv('./test.csv', encoding='utf-8')

