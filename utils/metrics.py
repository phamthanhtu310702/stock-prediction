from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score
from sklearn.metrics import f1_score

import torch


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy().flatten()
    labels = labels.cpu().numpy().flatten()

    f1 = f1_score(predicts, labels,average='micro')
    return {'F1 score': f1 }