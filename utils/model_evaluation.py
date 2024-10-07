import torch
import torch.nn as nn
from utils.metrics import get_node_classification_metrics
import numpy as np
import pandas as pd

def evaluate_model_node_classification(model: nn.Module, x_adaptor: nn.Module, y_adaptor: nn.Module,edges,edge_types, text_data, price_data, 
                                       dataloader, stocks_idx, criterion, device, threshold):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    model.eval()

    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = [], [], []
        pred_y_all = []
        for window_dates, label in dataloader:
            price_embeddings = x_adaptor(torch.tensor(price_data[: , torch.min(window_dates):torch.max(window_dates) + 1, :], dtype= torch.float32))
            logits = model(text_input=text_data, price_input=price_embeddings, windowtimes=window_dates, 
                        stock_name=list(stocks_idx.keys()),
                        edge_index=edges, num_edge_type=edge_types)

            adapted_label_list =[]
            for dates, logit, each_label in zip(window_dates, logits, label):
                aggregated_x = np.average(price_data[:, dates, :],axis=1)
                adapted_label = y_adaptor(x = torch.tensor(aggregated_x,dtype=torch.float32), y = logit, inverse = True)
                adapted_label_list.append(adapted_label)
                
                pred_y_all.append(
                        pd.DataFrame(
                            {
                                "pred": pd.Series(logit),
                                "label": pd.Series(each_label)
                            }
                        ).assign(date=pd.DatetimeIndex(56*[dates[-1]+1]))
                )
            
            adapted_label_list = torch.stack(adapted_label_list)
            loss = criterion(adapted_label_list, label)
            predicts =  adapted_label_list.sigmoid().to(device)
            
            evaluate_total_loss.append(loss.item())

            evaluate_y_trues.append(label)
            evaluate_y_predicts.append(predicts)

        evaluate_total_loss = np.mean(evaluate_total_loss)
        # evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        # evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)
        # evaluate_y_predicts = torch.where(evaluate_y_predicts < threshold, 0, 1)
        # evaluate_metrics = get_node_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)
        
        pred_y_all = pd.concat(pred_y_all)
        ic = pred_y_all.groupby("date").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
        return evaluate_total_loss, {'IC': ic }