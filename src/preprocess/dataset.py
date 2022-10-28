#!./env python
import torch
import os
import pandas as pd
import pickle
from transformers import BertTokenizer
from ..utils import print


def get_dataset(dataset, data_dir='./data', config=None):
    if dataset == 'imdb':
        return get_imdb(path=data_dir, config=config)
    else:
        raise KeyError('dataset: %s' % dataset)

def get_imdb(path='./data', filename='imdb_encoded.pt', config=None):
    ## load preprocessed data
    path = os.path.join(path, 'imdb')
    if os.path.exists(os.path.join(path, filename)):
        return torch.load(os.path.join(path, filename))

    ## preprocess raw data
    def map_review(text):
        return " ".join(text.strip().split("<br />"))

    def map_sentiment(text):
        return {"positive": 1, "negative": 0}.get(text)

    df = pd.read_csv(os.path.join(path,'IMDB.csv'))
    label_names = {i: l for i, l in enumerate(df['sentiment'].unique())}
    df['review'] = df['review'].apply(map_review)
    df['sentiment'] = df['sentiment'].apply(map_sentiment)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    encodings = tokenizer(df['review'].tolist(),  # Sentence to encode.
                          add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                          max_length=config.encoding_max_length,  # 
                          padding='max_length', # Pad short sentence
                          truncation=True, # Truncate long sentence
                          return_attention_mask=True,  # Construct attn. masks.
                          return_tensors='pt')  # Return pytorch tensors.
    labels = torch.tensor(df['sentiment'].tolist())

    data_dict = {'encodings': encodings,
                 'labels': labels,
                 'tokenizer': tokenizer,
                 'label_names': label_names}
    torch.save(data_dict, os.path.join(path, filename))
    
    return data_dict
