import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def cal(label, pred, sigmoid=False, threshold=0.5):
    # concatenate numpy array
    label = torch.cat(label)
    pred = torch.cat(pred)
    # sigmoid the pred
    if sigmoid:
        pred = torch.sigmoid(pred)
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    try:
        auc = roc_auc_score(label, pred)
    except:
        auc = 0.5
    # calculate precision, recall, f1 based label and pred
    pred = [1.0 if i >= threshold else 0.0 for i in pred]

    precision = precision_score(label, pred, zero_division=1)
    recall = recall_score(label, pred, zero_division=1)
    f1 = f1_score(label, pred, zero_division=1)

    return auc, precision, recall, f1


def cal_auc(label, pred):
    # concatenate numpy array
    label = torch.cat(label)
    pred = torch.cat(pred)
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    try:
        auc = roc_auc_score(label, pred)
    except:
        auc = 0.5
    return auc
