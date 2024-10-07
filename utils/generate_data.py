# Create a price file and a tweet file in form of npy file.
# Structure of npy file:
# Price: NxDxdp
# Text: NxDxdt

import tensorflow_hub as hub
import torch
import numpy as np
import pandas as pd


def process_text(file_path):
    text = pd.read_json(file_path, lines=True)
    
    #do some regrex here to clean the text before feeding it into the text encoder USE
    return text['text']

def USE_tweet_embedding(text, model):
    # number of texts in a day
    return model(text)


def GRU_text_embedding(emb, gru_model):
    one_day_tweet_emd = gru_model(emb)
    