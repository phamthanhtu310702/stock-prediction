import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Transformer import GraphTransformerConv
from model.modules import MLPClassifier
import pandas as pd
from typing import Union, List, Tuple, Dict, Text
class gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = nn.GRU(input_size = input_size, hidden_size=hidden_size, batch_first=True)
    def forward(self, inputs):
        full, last  = self.gru1(inputs)
        return full,last

class attn(nn.Module):
    def __init__(self,in_shape, out_shape ):
        super(attn, self).__init__()
        self.W1 = nn.Linear(in_shape, out_shape)
        self.W2 = nn.Linear(in_shape ,out_shape)
        self.V = nn.Linear(in_shape,1)
    def forward(self, full, last):
        score = self.V(F.tanh(self.W1(last) + self.W2(full)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * full
        context_vector =torch.sum(context_vector, dim=0) 
        return context_vector

class GNN(nn.Module):
    def __init__(self, n_in, n_hid, n_layers, n_heads, dropout, num_edge_type):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.gcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.init_fc = nn.Linear(n_in, n_hid)
        
        for i in range(n_layers):
            layer = GraphTransformerConv(
                n_in, n_hid // n_heads, heads=n_heads, dropout=dropout, 
                edge_type_self=0,
                num_edge_type=num_edge_type)
            self.gcs.append(layer)
            layer = nn.LayerNorm(n_hid)
            self.norms.append(layer)
    
    def forward(self, x, edge_index, edge_type):
        #convert to the iteratable 
        if edge_type is None:
            edge_type = [None] * len(edge_index)
        if not isinstance(edge_index, (list, tuple)):
            edge_index = [edge_index]
        if not isinstance(edge_type, (list, tuple)):
            edge_type = [edge_type]

        if edge_type is None:
            edge_type = [None] * len(edge_index)

        assert len(edge_index) == len(edge_type)
        
        # duplicate edge index for each layer
        if len(self.gcs) > len(edge_index):
            if len(edge_index) == 1:
                edge_index = [edge_index[0]] * len(self.gcs)
                edge_type = [edge_type[0]] * len(self.gcs)
            else:
                raise RuntimeError(
                    'Mismatch layer number gcs %d and edge_index %d!' % (
                        len(self.gcs), len(edge_index)))
        # if self.init_fc:
        #     x = self.init_fc(x)
        for conv, norm, ei, et in zip(self.gcs, self.norms, edge_index, edge_type):
            x = conv(x, ei, et)
            x = self.dropout(x)
            x = norm(x)
            x = torch.relu(x)
        return x

    
class StockNet(nn.Module):
    def __init__(self, gnn, nhid, dropout, n_layer,num_stocks):
        super(StockNet, self).__init__()
        
        self.num_stocks = num_stocks
        
        self.grup = gru(3,64)
        self.attnp = attn(64,64)
        self.grut = gru(64,64)
        self.attnt = attn(64,64)
        self.gru_tweet = gru(512,64)
        self.attn_tweet = attn(64,64)
        self.bilinear = nn.Bilinear(64,64,64)
        self.gnn = gnn
        
        #ff
        self.fc1 = nn.Linear(2*nhid, nhid)
        self.dropout = nn.Dropout(dropout)
        self.lyr_norm1 = nn.LayerNorm(normalized_shape=nhid)

        if n_layer == 1:
            pass
        elif n_layer == 2:
            self.fc2 = nn.Linear(nhid, nhid)
            self.lyr_norm2 = nn.LayerNorm(normalized_shape=nhid)
        else:
            raise NotImplementedError()
        self.n_layer = n_layer

        self.out   = nn.Linear(nhid, nhid)
        self.affinity_score = MLPClassifier(nhid, dropout=0.2)
        
        
        # create this outside the model like feature engineering
        # self.teacher_x = FeatureAdapter(x_dim=3)
        # self.teacher_y = LabelAdapter(x_dim =3)
        # self.meta_params = list(self.teacher_x.parameters()) + list(self.teacher_y.parameters())
        
    def forward(self, all_text_input: List, all_price_input: torch.Tensor, windowtimes, stock_name, edge_index, num_edge_type):
        batch_logits = [] # dates_numstock_binarylogits 
        for dates, text_input, price_input in zip(windowtimes, all_text_input, all_price_input):
            # all the stock_embeddings for the predicted date point.
            # stock_embedding = torch.zeros(self.num_stocks, 64)
            # tweet_embedding = torch.zeros(self.num_stocks, len(dates), 64)
            li = []
            for i in range(len(stock_name)):
                # x = self.teacher_x(price_input[i, dates, :])
                x = self.grup(price_input[i, : , :])
                x = self.attnp (*x).reshape((1,64))
                
                han_li1 = []
                for j in range(len(dates)):
                    text_embeddings = text_input[text_input['ID'].isin([stock_name[i]]) & text_input['Date'].isin([dates[j]])]['feature'].to_list()
                    if len(text_embeddings) > 0:
                        y = self.gru_tweet(torch.tensor(text_embeddings, dtype= torch.float32))
                        y = self.attn_tweet(*y).reshape((1,64))
                        # tweet_embedding[i][j] = y.clone()
                        han_li1.append(y)
                    else:
                        # tweet_embedding[i][j] = torch.zeros(1,64)
                        han_li1.append(torch.zeros(1,64))
                news_vector = torch.Tensor((len(dates),64))
                news_vector = torch.cat(han_li1)       
                text = self.grut(news_vector.reshape(len(dates),64))
                text = self.attnt(*text).reshape((1,64))
                combined = F.tanh(self.bilinear(text, x).reshape((1,64)))
                li.append(combined.reshape(1,64))
                
            #stock_embedding: numstock_64
            # out = gnn(stock_embedding, edges)
            #how to copy tensor properly
            ft_vec = torch.Tensor((self.num_stocks,64))
            ft_vec = torch.cat(li)
            out_0 = ft_vec
            out = self.gnn(ft_vec, edge_index, num_edge_type)
            out = torch.tanh(out)
            out = torch.cat([out_0, out], -1)
            
            #2-Feed-forward layer
            out = self.fc1(out)
            out = self.lyr_norm1(out)
            out = torch.relu(out)
            out = self.dropout(out)

            if self.n_layer == 1:
                pass
            elif self.n_layer == 2:
                out = self.fc2(out)
                out = self.lyr_norm2(out)
                out = torch.relu(out)
                out = self.dropout(out)
            else:
                raise NotImplementedError()

            out = self.affinity_score(out)
            logits  = torch.reshape(out, [-1])
            batch_logits.append(logits)
        
        return batch_logits

            