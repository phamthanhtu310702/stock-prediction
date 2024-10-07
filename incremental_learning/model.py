import copy
import os
import typing
from collections import defaultdict, OrderedDict
from typing import Dict, List, Union, Optional, Tuple

import numpy as np

from tqdm import tqdm
import pandas as pd
import torch
from torch import optim, nn
import torch.nn.functional as F
import higher
import higher_optim  # IMPORTANT, DO NOT DELETE

from model.adaptor_framework import LabelAdapter, FeatureAdapter
from incremental_learning import StockNet, GNN

class DoubleAdaptManager:
    """
    A meta-learning based incremental learning framework

    Args:
        model (torch.nn.Module): the stock trend forecasting model
        lr_model (float): the learning rate of the model
        lr_da (float): the learning rate of the data adapter
        lr_ma (float): the learning rate of the model adapter
        reg (float): regularization strength
        adapt_x (bool): whether to perform feature adaptation
        adapt_y (bool): whether to perform label adaptation
        first_order (bool): whether to adopt first-order approximation of MAML
        factor_num (int): the number of indicators at each time step of time-series inputs.
                    Otherwise, the same as :attr:`factor_num`
        x_dim (int): the total number of stock indicators
        need_permute (bool): whether to permute the last two dimensions of time-series input, specially for Alpha360.
        num_head (int): number of adaptation heads
        temperature (float): softmax temperature
        begin_valid_epoch (int): which epoch to begin validation. Set a moderate one to reduce training time.
    """
    def __init__(
        self,
        edges,
        num_edge_type,
        stock_dict: Dict,
        lr_model: float = 0.001,
        lr_da: float = 0.01,
        lr_ma: float = 0.001,
        lr_x: float = None,
        lr_y: float = None,
        online_lr: dict = None,
        weight_decay: float = 0,
        reg: float = 0.5,
        adapt_x: bool = True,
        adapt_y: bool = True,
        first_order: bool = True,
        factor_num: int = 6,
        x_dim: int = 360,
        need_permute: bool = True,
        num_head: int = 8,
        temperature: float = 10,
        begin_valid_epoch: int = 0,
    ):
        self.stock_dict = stock_dict
        self.lr_model = lr_model
        self.online_lr = online_lr
        self.reg = reg
        self.sigma = 1 ** 2 * 2
        self.factor_num = factor_num
        self.num_head = num_head
        self.temperature = temperature
        self.begin_valid_epoch = begin_valid_epoch
        self.first_order = first_order
        self.edges = edges
        self.num_edge_type = num_edge_type
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.num_stocks = len(stock_dict)
        self.stock_name = list(stock_dict.values())
        
        self.x_adaptor = FeatureAdapter(in_dim=3)
        self.y_adaptor = LabelAdapter(x_dim= 3, hid_dim= 32, edges= self.edges)
        self.gnn = GNN(n_in=64, n_hid=64, n_layers=6, n_heads=8, dropout=0.2, num_edge_type=self.num_edge_type)
        self.model = self.init_StockNet(num_stocks=self.num_stocks)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=lr_model, weight_decay=weight_decay)
        self.opt = self._init_meta_optimizer(lr_x = lr_x, lr_y = lr_y)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.fmodel = None

    def init_StockNet(self, num_stocks):
        
        model = StockNet(gnn = self.gnn, nhid=64, dropout=0.2, n_layer=2, num_stocks =num_stocks)
        
        return model

    def _init_meta_optimizer(self, lr_x=0.01, lr_y=0.01):
        """ NOTE: the optimizer of the model adapter is self.framework.opt """
        
        return optim.Adam([{'params': self.x_adaptor.parameters(), 'lr': lr_x},
                            {'params': self.y_adaptor.parameters(), 'lr': lr_y},])

    
    def load_state_dicts(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.x_adaptor.load_state_dict(torch.load(path))
        self.y_adaptor.load_state_dict(torch.load(path))
    
    def override_online_lr_(self):
        if self.online_lr is not None:
            if 'lr_model' in self.online_lr:
                self.lr_model = self.online_lr['lr_model']
            if 'lr_ma' in self.online_lr:
                self.framework.opt.param_groups[0]['lr'] = self.online_lr['lr_ma']
            if 'lr_da' in self.online_lr:
                self.opt.param_groups[0]['lr'] = self.online_lr['lr_da']
            else:
                if 'lr_x' in self.online_lr:
                    self.opt.param_groups[0]['lr'] = self.online_lr['lr_x']
                if 'lr_y' in self.online_lr:
                    self.opt.param_groups[1]['lr'] = self.online_lr['lr_y']

    def _run_task(self, meta_input_price: Dict[str, Union[pd.Index, torch.Tensor]],
                  meta_input_tweet: Dict[str, Union[pd.Index, torch.Tensor]], 
                  labels,
                  date_ranges: List,
                  phase: str):

        self.model_opt.zero_grad()
        self.opt.zero_grad()

        if 'train' in phase:
            """ Incremental data adaptation & Model adaptation """
            price_train_tasks = meta_input_price["train"]
            tweet_train_tasks = meta_input_tweet["train"]
            labels_train_tasks = labels["train"]
            
            with higher.innerloop_ctx(
                self.model,
                self.model_opt,
                copy_initial_weights=False,
                track_higher_grads=not self.first_order,
                override={'lr': [self.lr_model]}
            ) as (fmodel, diffopt):
                with torch.backends.cudnn.flags(enabled=self.first_order):
        
                    price_embeddings = self.x_adaptor(torch.stack(price_train_tasks))
                    y_hat= fmodel(tweet_train_tasks, price_embeddings, date_ranges['train'], self.stock_name, self.edges, self.num_edge_type)
                    y_hat = torch.stack(y_hat)

                    adapted_label_train_list =[]
                    raw_y = labels_train_tasks
                    
                    for prices, label in zip(price_train_tasks, labels_train_tasks):
                    
                        aggregated_x = np.average(prices,axis=1)
                        adapted_label = self.y_adaptor(x = torch.tensor(aggregated_x,dtype=torch.float32), y = label, inverse = False)
                        adapted_label_train_list.append(adapted_label)
                    
                    adapted_label_train_list = torch.stack(adapted_label_train_list) 
                    loss2 = self.criterion(y_hat, adapted_label_train_list)
                    loss_y = F.mse_loss(adapted_label_train_list, torch.stack(raw_y))
                    loss_y = loss_y * self.reg
                    loss = loss2 + loss_y
                    diffopt.step(loss)
                    # this model is updated and use for the inference and online_training on the test data
            self.fmodel = fmodel
        
        """ Online inference """
        if 'test' in phase:
            price_test_tasks = meta_input_price["test"]
            tweet_test_tasks = meta_input_tweet["test"]
            labels_test_tasks = labels["test"]
            
            X_test_adapted = self.x_adaptor(torch.stack(price_test_tasks))
            
            preds = self.fmodel(tweet_test_tasks, X_test_adapted, date_ranges['test'], self.stock_name, self.edges, self.num_edge_type)
            
            adapted_label_list =[]
            
            for prices, pred in zip(price_test_tasks, preds):

                aggregated_x = np.average(prices,axis=1)
                adapted_label = self.y_adaptor(x = torch.tensor(aggregated_x,dtype=torch.float32), y = pred, inverse = True)
                adapted_label_list.append(adapted_label)
                
            preds = torch.stack(adapted_label_list)
            
            """ Optimization of meta-learners """
            loss = self.criterion(preds, torch.stack(labels_test_tasks))
            loss.backward()
            if self.adapt_x or self.adapt_y:
                self.opt.step()
            self.model_opt.step()
            return preds
    
    
    def _run_epoch(self, phase: str, price_tasks, tweet_tasks, label_tasks, date_task_dict,
                   tqdm_show: bool=False):
        # fast apdation for online training
        pred_y_all = []
        self.override_online_lr_()

        self.phase = phase

        preds = self._run_task(price_tasks, tweet_tasks, label_tasks, date_task_dict, phase)
        
        if phase != "train":
            for pred , label, dates in zip(preds, label_tasks["test"],date_task_dict['test']):
                pred_y_all.append(
                    pd.DataFrame(
                        {
                            "pred": pd.Series(pred),
                            "label": pd.Series(label.reshape(56)),
                        }
                    ).assign(date=pd.DatetimeIndex(56*[dates[-1]+1]))
                )
        if phase != "train":
            pred_y_all = pd.concat(pred_y_all)
            ic = pred_y_all.groupby("date").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
        return pred_y_all, ic

    def inference(self, price_tasks, tweet_tasks, label_tasks, date_task_dict):
        """
        Perform incremental learning on the test set.

        Returns:
            pd.DataFrame:
                the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
                the col named 'pred' contains the predictions of the model;
                the col named 'label' contains the ground-truth labels which have been preprocessed and may not be the raw.
        """
        self.model.train()
        self.model.to(self.framework.device)
        pred_y_all, ic = self._run_epoch("online", price_tasks, tweet_tasks, label_tasks, date_task_dict, tqdm_show=True)
        return pred_y_all, ic