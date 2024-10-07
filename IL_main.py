# Implement IL with GNN model
import os
import time 
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict, Text
import torch
import torch.nn as nn
from incremental_learning.model import DoubleAdaptManager
from model.adaptor_framework import LabelAdapter, FeatureAdapter
from model import StockNet, GNN
from datetime import timedelta

from ast import literal_eval

class OnlineInference:
    def __init__(
        self,
        reload_path,
        num_stocks,
        new_edge_types,
        data_dir="stock_dataset",
        market="NYSE",
        alpha=360,
        x_dim=None,
        step=20,
        model_name="GNN",
        lr=0.001,
        lr_ma=None,
        lr_da=0.01,
        lr_x=None,
        lr_y=None,
        online_lr: dict = None,
        reg=0.5,
        weight_decay=0,
        num_head=8,
        tau=10,
        first_order=True,
        adapt_x=True,
        adapt_y=True,
        tag=None,):
        
        self.data_dir = data_dir
        self.market = market
        self.step = step
        self.model_name = model_name  # downstream forecasting models' type
        self.alpha = alpha
        self.tag = tag
        if self.tag is None:
            self.tag = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        self.lr_da = lr_da
        self.lr_ma = lr if lr_ma is None else lr_ma
        self.lr_x = lr_x
        self.lr_y = lr_y
        if online_lr is not None and 'lr' in online_lr:
            online_lr['lr_model'] = online_lr['lr']
        self.online_lr = online_lr
        self.num_head = num_head
        self.temperature = tau
        self.first_order = first_order
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.weight_decay = weight_decay
        self.not_sequence = self.model_name in ["MLP", 'Linear'] and self.alpha == 158
        self.factor_num = 6 if self.alpha == 360 else 20
        self.x_dim = x_dim if x_dim else (360 if self.alpha == 360 else 20 * 20)
        self.reload_path = reload_path
        
        self.stock_dict = pd.read_csv("all_stock_idx")
        self.edges_graph = pd.read_csv("edges.csv")

        self.framework = self.init_framework()
        
    
    def init_framework(self):
        """Init the Incremental framework"""
        framework = DoubleAdaptManager(edges=self.edges_graph, stock_dict = self.stock_dict, x_dim=self.x_dim, lr_model=self.lr, weight_decay=self.weight_decay,
                                               first_order=self.first_order, begin_valid_epoch=0, factor_num=self.factor_num,
                                               lr_da=self.lr_da, lr_ma=self.lr_ma, online_lr=self.online_lr,
                                               lr_x=self.lr_x, lr_y=self.lr_y,
                                               adapt_x=self.adapt_x, adapt_y=self.adapt_y, reg=self.reg,
                                               num_head=self.num_head, temperature=self.temperature)
        
        return framework
    
    def init_StockNet(self, num_stocks, saved_model_path):
        model = StockNet(gnn = self.gnn, nhid=64, dropout=0.2, n_layer=2, num_stocks =num_stocks)
        model.load_state_dict(torch.load(saved_model_path))
        return model

        
        
    def load_incremental_data(self, path):
        """
        *Note: this function just loads the preprocessed data, do not process data here
        Returns:
            pd.DataFrame:
                the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
                the col named 'feature' contains the stock feature vectors;
                the col named 'label' contains the ground-truth labels.
        """
        
        df_text = pd.read_csv('il_data/price.csv', converters={'feature': literal_eval})
        df_price= pd.read_csv('il_data/price.csv', converters={'feature': literal_eval})
        
        return df_price, df_text
    

    def preprocess_tweet(self, data: pd.DataFrame, date_range_dict: Dict):
        """
            seg: { 
                train : { START_DATE, END_DATE},
                test : {START_DATE, END_DATE}
            }
        """
        
        tweet_tasks = {}
        stock_idx = list(self.stock_dict.values())
        
        for phase, date_ranges in date_range_dict:
            tweet_tasks[phase] = []
            for date_range in date_ranges:
                tweets  = data[data['ID'].isin(stock_idx) & data['Date'].isin(date_range)]
                tweet_tasks[phase].append(tweets)

        return tweet_tasks
    
    
    def get_stock_data_based_idx(data: pd.DataFrame, stock_idx: str):
        
        stock_data = data[data.index.isin([stock_idx], level=1)]
        
        return stock_data
    
    def get_text_embeddings_by_stock_idx(data: pd.DataFrame, stock_idx: str, seg):
        
        text_embeddings = data[data.index.isin([stock_idx], level=1)].loc(axis=0)[seg[0] : seg[1]]
        
        return text_embeddings
    
    def convert_to_Tensor(data: pd.DataFrame):
        
        return torch.tensor(data['feature'],  dtype=torch.float32)
        
    def create_dates_tasks(seg):
        date_task_dict = {}
        for k, v in seg.items():
            dates_range = pd.date_range(v[0], v[1])
            date_task_dict[k] = []
            for date in dates_range:
                start_date = (date - timedelta(days=8)).strftime("%Y-%m-%d")
                end_date = (date).strftime("%Y-%m-%d")
                dates = pd.date_range(start_date, end_date)
                date_task_dict[k].append(dates)

        return date_task_dict
    
    def convert_dataframe_to_tensor(self, data: pd.DataFrame, date_range):
        new_vec = []
        labels = []
        
        for k, v in self.stock_dict.items():
            tensor = data[data['ID'].isin([v]) & data['Date'].isin(date_range)]
            feature_vec = tensor['feature'].to_list()
            label = tensor[tensor['Date'] == date_range[-1]]['label'].to_list()
            
            new_vec.append(torch.tensor(feature_vec, dtype= torch.float32))
            labels.append(torch.tensor(label, dtype= torch.float32))
        
        new_vec = torch.stack(new_vec)
        labels = torch.stack(labels)
        return new_vec, labels
    
    def preprocess_price(self, data: pd.DataFrame, date_range_dict):
        
        price_tasks = {}
        label_tasks = {}
        for phase, all_date_ranges in date_range_dict.items():
            price_tasks[phase] = []
            label_tasks[phase] = []
            
            for each_date_range in all_date_ranges:
                new_vec, labels = self.convert_dataframe_to_tensor(data, each_date_range)
                price_tasks[phase].append(new_vec)
                label_tasks[phase].append(labels)
        
        return price_tasks, label_tasks
        
    def online_inference(self, tweet_embedding, price_embedding):
        """ 
            Deployed model with updated parameters for daily inference
            Query data before the current date.
        """
        pred_y = self.framework.model(tweet_embedding, price_embedding, self.edges_graph)

        return pred_y
    
    def save_results(pred_y_all):
        pass
    
    def online_training(self, seg: Dict, reload_path: str):
        """Return the predictions of stock trends"""
        """equivalent to online_training func in original project"""
        self.framework.load_state_dict(torch.load(reload_path))
        
        """load current data"""
        
        date_task_dict = self.create_dates_tasks(seg)
        price, tweet = self.load_incremental_data(path='stocknet-dataset/price/raw/')
        
        price_tasks, label_tasks = self.preprocess_price(price, date_task_dict)
        
        tweet_tasks = self.preprocess_tweet(tweet, date_task_dict)
        
        
        pred_y_all = self.framework.inference(price_tasks, tweet_tasks, label_tasks, date_task_dict)
        
        self.save_results(pred_y_all)

            
