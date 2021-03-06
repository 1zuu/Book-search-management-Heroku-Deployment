import os
import re
import json
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

import tensorflow as tf
logging.getLogger('tensorflow').disabled = True
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from variables import*

def lemmatization(lemmatizer,sentence):
    '''
        Lemmatize texts in the terms
    '''
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = list(dict.fromkeys(lem))

    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    '''
        Remove stop words in texts in the terms
    '''
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(description):
    '''
        Text preprocess on term text using above functions
    '''
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    description = description.lower()
    remove_punc = tokenizer.tokenize(description) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_punc if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_description = ' '.join(remove_stop)
    return updated_description

def preprocessed_data(descriptions):
    '''
        Preprocess entire terms
    '''
    updated_descriptions = []
    if isinstance(descriptions, np.ndarray) or isinstance(descriptions, list):
        for description in descriptions:
            updated_description = preprocess_one(description)
            updated_descriptions.append(updated_description)
    elif isinstance(descriptions, np.str_)  or isinstance(descriptions, str):
        updated_descriptions = [preprocess_one(descriptions)]

    return np.array(updated_descriptions)

def load_json_file(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def load_json_data():
    json_files = os.listdir(json_dir)
    json_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    json_files = [os.path.join(json_dir, json_file) for json_file in json_files]
    return np.array([load_json_file(json_file) for json_file in json_files if json_file.endswith('.json')])

def get_Data():
    '''
        Get data from database
    '''
    df = pd.read_csv(data_path)
    df_response = df[['Category', 'Description', 'Book_title' ,'Author' ,'ISBN-10' ,'ISBN-13', 'Cover_link']] 
    df_response = df_response.fillna('NULL')
    prices = load_json_data()
    assert len(df_response) == len(prices), "Dataframe and Prices are not same size"
    return df_response, prices

def tokenizer_save_and_load(tokenizer=None):
    if tokenizer:
        file_ = open(tokenizer_weights,'wb')
        pickle.dump(tokenizer, file_, protocol=pickle.HIGHEST_PROTOCOL)
        file_.close()
    else:
        assert os.path.exists(tokenizer_weights), "Tokenizer Weights doesn't exists. Please Save before load."
        file_ = open(tokenizer_weights,'rb')
        tokenizer = pickle.load(file_)
        file_.close()
        return tokenizer

def prediction_data(description):
    tokenizer = tokenizer_save_and_load()
    processed_description = preprocess_one(description)
    x_seq = tokenizer.texts_to_sequences([processed_description])
    x_pad = pad_sequences(x_seq, maxlen=max_length, padding=padding, truncating=trunc_type)
    return x_pad

def load_category_df(df_response, prices, pred_category):
    df_response['Category'] = df_response['Category'].str.strip()
    df_response['Category'] = df_response['Category'].str.lower()
    price_response = prices[df_response['Category'] == pred_category]
    df_response = df_response[df_response['Category'] == pred_category]
    return df_response, price_response

def reform_prices(sample):
    # new_price_details = {}
    # price_list = []
    # websites = []
    # details = []
    # for key, value in sample.items():
    #     if set(['price', 'link']) == set(list(value.keys())):
    #         value["price"] = value["price"].replace("$", "")
    #         value["price"] = value["price"].strip()
    #         try:
    #             if value["price"] == "NULL":
    #                 value["price"] = 0
    #             else:
    #                 value["price"] = float(value["price"])
    #             price_ = value["price"]
    #             price_list.append(price_)
    #             value["price"] = '$'+str(value["price"]).strip()
    #             websites.append(key)
    #             details.append(value)
    #         except:
    #             pass
    # price_list = np.array(price_list)
    # price_order = np.argsort(price_list)
    # for i in price_order:
    #     new_price_details[websites[i]] = details[i]
    # return new_price_details
    
    price_details = []
    new_price_details = []
    price_list = []
    websites = []
    details = []
    for key, value in sample.items():
        if set(['price', 'link']) == set(list(value.keys())):
            value["price"] = value["price"].replace("$", "")
            value["price"] = value["price"].strip()
            try:
                if value["price"] == "NULL":
                    value["price"] = 0
                else:
                    value["price"] = float(value["price"])
                price_ = value["price"]
                price_list.append(price_)
                value["price"] = '$'+str(value["price"]).strip()
                value["website"] = key.strip()
                price_details.append(value)
            except:
                pass
    price_list = np.array(price_list)
    price_order = np.argsort(price_list)
    for i in price_order:
        new_price_details.append(price_details[i])
    return new_price_details


    # new_price_details = []
    # for key, value in sample.items():
    #     if set(['price', 'link']) == set(list(value.keys())):
    #         value["price"] = value["price"].replace("$", "")
    #         value["price"] = value["price"].strip() if value["price"] != "NULL" else 0
    #         value["price"] = '$'+ value["price"].strip()
    #     value["website"] = key.strip()
    #     new_price_details.append(value)
    # return new_price_details