import pandas as pd
import torch
import json
import ast
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForMultipleChoice
from tqdm import tqdm
from itertools import chain
import argparse

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, default='./MC')
parser.add_argument("--config_path", type=str, default='./MC/config.json')
parser.add_argument("--context_path", type=str, default="/mnt/nas4/weiwei/ADL/proj2/data/context.json")
parser.add_argument("--test_path", type=str, default='/mnt/nas4/weiwei/ADL/proj2/data/SWAG/test.csv')
parser.add_argument('--device', type=str, default='cpu')
opt = parser.parse_args()

# 設定 device (opt.device = 'cpu' or 'cuda:X')
if opt.device[:4] == 'cuda':
    gpu_id = opt.device[-1]
    #os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    torch.cuda.set_device(opt.device)
    device = torch.device(opt.device)
else:
    print('device: cpu')
    device = torch.device('cpu')

# load fine-tuned model & tokenizer
config = AutoConfig.from_pretrained(opt.config_path)
tokenizer = AutoTokenizer.from_pretrained(opt.checkpoint_dir)
model = AutoModelForMultipleChoice.from_pretrained(opt.checkpoint_dir).to(device)

# load context.json & SWAG/test.csv
with open(opt.context_path, 'r') as f:
    context = json.load(f)

ending_names = [f"ending{i}" for i in range(4)]

test = pd.read_csv(opt.test_path)
test['paragraphs'] = test['paragraphs'].map(ast.literal_eval)

for i, row in tqdm(test.iterrows(), total=len(test)):
    question = [row['sent2']]
    
    second_sentences = [
        [f"{header} {row[end]}" for end in ending_names] for i, header in enumerate(question)
    ]
    second_sentences = list(chain(*second_sentences))
    
    # tokenize
    tokenized_examples = tokenizer(
            second_sentences,
            truncation=True,
            max_length=384,
            padding="max_length",
        )
    
    x = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    x['input_ids'] = torch.tensor(x['input_ids']).to(device)
    x['token_type_ids'] = torch.tensor(x['token_type_ids']).to(device)
    x['attention_mask'] = torch.tensor(x['attention_mask']).to(device)
    
    # predict
    out = model(**x).logits.cpu().detach().numpy()
    pred = np.argmax(out, axis=1)[0]
    
    pred_context = row['paragraphs'][pred]    
    test.at[i, 'context'] = context[pred_context]
    
test = test.rename(columns={'sent2': 'question'})
test = test.drop(['sent1', 'paragraphs', 'ending0', 'ending1', 'ending2', 'ending3'], axis='columns')

test.to_csv('./QA/to_predict.csv', encoding='utf-8')
