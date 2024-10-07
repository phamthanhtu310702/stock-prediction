import json
import numpy as np
import pandas as pd
import torch
import os
    
def build_wiki_relation(market_name, connection_file, tic_wiki_file,
                        sel_path_file):
    
    stock_path= 'C:/Users/Admin/Documents/stock-prediction/stocknet-dataset/price/preprocessed'
    available_stocks_data = os.listdir(stock_path)
    available_stocks = []
    for i in available_stocks_data:
        available_stocks.append(i.split('.')[0])
        
    # get valid tickers
    tickers = pd.read_csv('data/relation/wikidata/NYSE_wiki.csv',header= None)
    stocks_list =  tickers[tickers[1] != 'unknown'][0].tolist()
    
    count = 0
    available_stocks_match = []
    for i in available_stocks:
        if i in stocks_list:
            available_stocks_match.append(i)
            count +=1
    available_stocks_data = tickers[tickers[0].isin(available_stocks_match)]
    
    
    wikiid_ticind_dic = {}
    start = 0
    for ind, tw in available_stocks_data.iterrows():
        if not tw[1] == 'unknown':
            wikiid_ticind_dic[tw[1]] = start
            start +=1
    print('#tickers aligned:', len(wikiid_ticind_dic))
    
    sel_paths = np.genfromtxt(sel_path_file, dtype=str, delimiter=' ',
                              skip_header=False)
    sel_paths = set(sel_paths[:, 0])
    
    
    # readin connections - relations file
    with open(connection_file, 'r') as fin:
        connections = json.load(fin)
        
    occur_paths = set()
    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in sel_paths:
                    occur_paths.add(path_key)
    
    valid_path_index = {}
    for ind, path in enumerate(occur_paths):
        valid_path_index[path] = ind
        
    available_stocks_match_idx = {}
    for idx, stock in enumerate(available_stocks_match):
        available_stocks_match_idx[stock] = idx
    
    edge_lst =[]
    for src, dst in connections.items():
        if src in wikiid_ticind_dic.keys():
            for tar_item, paths in dst.items():
                if tar_item in wikiid_ticind_dic.keys():
                    for p in paths:
                        path_key = '_'.join(p)
                        if path_key in valid_path_index.keys():
                            #wiki_relation_embedding[wikiid_ticind_dic[src]][wikiid_ticind_dic[tar_item]][valid_path_index[path_key]] = 1
                            new_edge = {'src': wikiid_ticind_dic[src],
                                        'dst': wikiid_ticind_dic[tar_item],
                                        'edge_type': valid_path_index[path_key]}
                            edge_lst.append(new_edge)

    edge_construction = pd.DataFrame(edge_lst, columns=['src', 'dst', 'edge_type'])
    
    # review the construction edge Tensor in the previous project - Fraud detection project
    edges = torch.tensor(edge_construction[['src','dst']].to_numpy(), dtype=torch.long)
    edge_types= torch.tensor(edge_construction['edge_type'].to_numpy(), dtype=torch.int32)
    
    return edges, edge_types, wikiid_ticind_dic, available_stocks_match_idx
    
    # add self-relation
    
    # save wikiid_ticind_dic and valid_path_index as graph metada for later inference