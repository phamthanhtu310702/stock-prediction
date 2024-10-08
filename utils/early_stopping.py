
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, model_name: str = None):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param save_model_name: str, save model name
        :param model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.save_model_path = os.path.join(save_model_folder, "stock_predictor.pkl")
        self.save_x_adaptor_path = os.path.join(save_model_folder, "x_adaptor.pkl")
        self.save_y_adaptor_path = os.path.join(save_model_folder, "y_adaptor.pkl")
        self.model_name = model_name

    def step(self, metrics: list, model: nn.Module, x_adapter: nn.Module, y_adaptor: nn.Module):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: nn.Module
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model, x_adapter, y_adaptor)
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module,  x_adapter: nn.Module, y_adaptor: nn.Module):
        """
        saves model at self.save_model_path
        :param model: nn.Module
        :return:
        """
        torch.save(model.state_dict(), self.save_model_path)
        torch.save(x_adapter.state_dict(), self.save_x_adaptor_path)
        torch.save(y_adaptor.state_dict(), self.save_y_adaptor_path)

    def load_checkpoint(self, model: nn.Module,  x_adapter: nn.Module, y_adaptor: nn.Module ,map_location: str = None):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        model.load_state_dict(torch.load(self.save_model_path, map_location=map_location))
        x_adapter.load_state_dict(torch.load(self.save_x_adaptor_path, map_location=map_location))
        y_adaptor.load_state_dict(torch.load(self.save_y_adaptor_path, map_location=map_location))

def criterion(prediction_dict, labels, model, mode, device):

    # for key, value in prediction_dict.items():
    #     if key != 'root_embedding' and key != 'group' and key != 'dev':
    #         prediction_dict[key] = value[labels > -1]

    labels = labels[labels > -1]
    logits = prediction_dict['logits']

    loss_classify = F.binary_cross_entropy_with_logits(
        logits, labels, reduction='none')
    loss_classify = torch.mean(loss_classify)

    loss = loss_classify.clone()
    
    if mode == 'sad' and model != None:
        loss_anomaly = torch.Tensor(0).to(device)
        loss_supc = torch.Tensor(0).to(device)
        alpha = 1e-1
        beta = 1e-3
        loss_anomaly = model.gdn.dev_loss(torch.squeeze(labels), torch.squeeze(prediction_dict['anom_score']), torch.squeeze(prediction_dict['time'].to(device)))
        loss_supc = model.suploss(prediction_dict['root_embedding'], prediction_dict['group'], prediction_dict['dev'])
        loss += alpha * loss_anomaly + beta * loss_supc

    return loss, loss_classify