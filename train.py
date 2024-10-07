import torch
from utils.dataloader import DateDataset
from torch.utils.data import DataLoader
from utils.generate_relation import build_wiki_relation
from model.StockNet import StockNet, GNN
from utils.early_stopping import EarlyStopping
from utils.model_evaluation import evaluate_model_node_classification
from utils.metrics import get_node_classification_metrics
from model.adaptor_framework import FeatureAdapter, LabelAdapter
import os
import json
import numpy as np
import torch.nn as nn
import shutil
import logging
import time
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(tic_wiki_file='data/relation/wikidata/NYSE_wiki.csv', 
         sel_path_file='data/relation/wikidata/selected_wiki_connections.csv', 
         epoch_size = 1,
         threshold = 0.6):
    # with open(available_stocks_path, 'r') as fin:
    #     available_stocks_data = json.load(fin)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./logs/", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./logs/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    edges, edge_types, ticker_idx, stocks_idx = build_wiki_relation(market_name = 'NYSE', 
                                                    connection_file= 'data/relation/wikidata/NYSE_connections.json',
                                                    tic_wiki_file = tic_wiki_file,
                                                    sel_path_file =sel_path_file )
    edges = edges.t().contiguous()
    new_edge_types = edge_types.unique().tolist()
    encode_to_new = dict((e,i) for i, e in enumerate(new_edge_types))
    f = lambda x: encode_to_new[x]
    f = np.vectorize(f)
    edge_types = f(edge_types)
    edge_types = torch.LongTensor(edge_types)
    
    num_stocks = len(stocks_idx)
    unique_days = []
    for name in stocks_idx.keys():
        path = 'stocknet-dataset/tweet/raw/' +  name
        days = os.listdir(path)
        for day in days:
            if day not in unique_days:
                unique_days.append(day)
    unique_days.sort()            
    
    # change it to pandas.Dataframe for querying instead of using numpy.array with unique_days_index_inverse
    unique_days_index_inverse = {}
    for i, day in enumerate(unique_days):
        unique_days_index_inverse[i] = day
    
    #load training data
    all_price_embedding = np.load('price_embeddings.npy')
    
    with open("tweet_embeddings.json", 'r') as fin:
        data = json.load(fin)

    #load label data
    # create simulated labeled data
    customed_train_labels = torch.empty((16,56)).random_(2)
    customed_val_labels = torch.empty((16,56)).random_(2)
    customed_test_labels = torch.empty((16,56)).random_(2)
    
    #dates split
    dates_index = list(unique_days_index_inverse.keys())
    dates = len(unique_days)
    train_range = dates*0.7
    val_range = dates*0.85
    
    #split the label data as well
    
    # window_dates = DateDataset(torch.tensor(range(len(unique_days))),window= 8)
    #train-val-test dataset
    window_dates = DateDataset(torch.tensor(dates_index[:16]),window= 8, labels=customed_train_labels)
    # val_window_dates =  DateDataset(torch.tensor(unique_days[int(train_range)+1 : int(val_range)]),window= 8, labels=customed_val_labels)
    # test_window_dates =  DateDataset(torch.tensor(unique_days[int(val_range)+1 : ]),window= 8, labels=customed_test_labels)
    
    #code for debug-testing
    val_window_dates =  DateDataset(torch.tensor(dates_index[16 : 32 ]),window= 8, labels=customed_val_labels)
    test_window_dates =  DateDataset(torch.tensor(dates_index[32 : 48]),window= 8, labels=customed_test_labels)
    
    dl = DataLoader(window_dates, batch_size= 64)
    val_dl = DataLoader(val_window_dates, batch_size= 64)
    test_dl = DataLoader(test_window_dates, batch_size= 64)
    
    
    gnn = GNN(n_in=64, n_hid=64, n_layers=6, n_heads=8, dropout=0.2, num_edge_type=len(new_edge_types))
    model = StockNet(gnn = gnn, nhid=64, dropout=0.2, n_layer=2, num_stocks =num_stocks, unique_days_index_inverse = unique_days_index_inverse)
    params = list(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    y_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(params, lr= 0.001)
    
    # load data adaptor
    teacher_x = FeatureAdapter(in_dim=3)
    teacher_y = LabelAdapter(x_dim= 3, hid_dim= 32, edges= edges)
    meta_params = list(teacher_x.parameters()) + list(teacher_y.parameters())
    meta_opt = torch.optim.Adam(meta_params, lr=0.01)
    
    model_name = 'stock-predictor'
    save_model_folder = f"./saved_models/{model_name}/"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    early_stopping = EarlyStopping(patience=20, save_model_folder=save_model_folder, model_name=model_name)
    
    torch.autograd.set_detect_anomaly(True)
    for _ in range(1):
        train_total_loss, train_y_trues, train_y_predicts = [], [], []
        for training_dates, label in dl:
            # find min and max idx day in tensor
            price_embeddings = teacher_x(torch.tensor(all_price_embedding[: , torch.min(training_dates):torch.max(training_dates) + 1, :], dtype= torch.float32))
            logits = model(text_input=data, price_input=price_embeddings, windowtimes=training_dates, 
                        stock_name=list(stocks_idx.keys()),
                        edge_index=edges, num_edge_type=edge_types)
            
            raw_y = label
            adapted_label_list =[]
            for dates, labels in zip(training_dates, label):
                aggregated_x = np.average(all_price_embedding[:, dates, :],axis=1)
                adapted_label = teacher_y(x = torch.tensor(aggregated_x,dtype=torch.float32), y = labels, inverse = False)
                adapted_label_list.append(adapted_label)
                
            
            logits = torch.stack(logits)
            adapted_label_list = torch.stack(adapted_label_list)
            
            loss2 = y_criterion(raw_y, adapted_label_list)
            loss = criterion(logits, adapted_label_list)
            predicts =  logits.sigmoid().to(device)
            train_total_loss.append(loss.item())
            train_y_trues.append(label)
            train_y_predicts.append(predicts)

            loss.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            meta_opt.step()

            print('success!')
            break
        
        train_total_loss = np.mean(train_total_loss)
        train_y_trues = torch.cat(train_y_trues, dim=0)
        train_y_predicts = torch.cat(train_y_predicts, dim=0)
        #use where to convert the predicts to 0,1
        train_y_predicts = torch.where(train_y_predicts < threshold, 0, 1)
        
        # train_metrics = get_node_classification_metrics(predicts=train_y_predicts, labels=train_y_trues)  
        
        val_total_loss, val_metrics = evaluate_model_node_classification(model=model,
                                                                        x_adaptor=teacher_x,
                                                                        y_adaptor=teacher_y,
                                                                        edges = edges,
                                                                        edge_types = edge_types,
                                                                        text_data = data,
                                                                        price_data=all_price_embedding,
                                                                        criterion = criterion,
                                                                        dataloader=val_dl,
                                                                        stocks_idx = stocks_idx,
                                                                        device= device,
                                                                        threshold = threshold)
        logger.info(f'train loss: {train_total_loss:.4f}')
        # for metric_name in train_metrics.keys():
        #     logger.info(f'train {metric_name}, {train_metrics[metric_name]:.4f}')
        
        logger.info(f'validate loss: {val_total_loss:.4f}')
        for metric_name in val_metrics.keys():
            logger.info(f'validate {metric_name}, {val_metrics[metric_name]:.4f}')

        val_metric_indicator = []
        for metric_name in val_metrics.keys():
            val_metric_indicator.append((metric_name, val_metrics[metric_name], True))
        early_stop = early_stopping.step(val_metric_indicator, model, teacher_x, teacher_y)

        if early_stop:
            break
        
    early_stopping.load_checkpoint(model, teacher_x, teacher_y)
    
    test_total_loss, test_metrics = evaluate_model_node_classification(model=model,
                                                                        x_adaptor=teacher_x,
                                                                        y_adaptor=teacher_y,
                                                                        edges = edges,
                                                                        edge_types = edge_types,
                                                                        text_data = data,
                                                                        price_data=all_price_embedding,
                                                                        criterion = criterion,
                                                                        dataloader=test_dl,
                                                                        stocks_idx = stocks_idx,
                                                                        device= device,
                                                                        threshold = threshold)
    val_metric_dict, test_metric_dict = {}, {}
    logger.info(f'validate loss: {val_total_loss:.4f}')
    for metric_name in val_metrics.keys():
        val_metric = val_metrics[metric_name]
        logger.info(f'validate {metric_name}: {val_metric:.4f}')
        val_metric_dict[metric_name] = val_metric

    logger.info(f'test loss: {test_total_loss:.4f}')
    for metric_name in test_metrics.keys():
        test_metric = test_metrics[metric_name]
        logger.info(f'test {metric_name}: {test_metric:.4f}')
        test_metric_dict[metric_name] = test_metric

    result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
        }
    

    save_result_folder = f"./saved_results/"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, "result.json")

    with open(save_result_path, 'w') as file:
        json.dump(result_json, file)

            
    #save graph metadata for inference
if __name__ == '__main__':
    main()