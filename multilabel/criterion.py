import torch.nn as nn
from utils import calculate_accuracy


class FashionLoss(nn.Module):

    def __init__(self):
        super(FashionLoss, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y):
        y = y.float()
        loss = self.BCE(y_pred, y)
        return loss


class FashionAccuracy(nn.Module):

    def __init__(self):
        super(FashionAccuracy, self).__init__()

    def forward(self, y_pred, y):
        y_pred_1, y_1 = y_pred[:, :7], y[:, :7]
        acc_1 = calculate_accuracy(y_pred_1, y_1)
        y_pred_2, y_2 = y_pred[:, 7:10], y[:, 7:10]
        acc_2 = calculate_accuracy(y_pred_2, y_2)
        y_pred_3, y_3 = y_pred[:, 10:13], y[:, 10:13]
        acc_3 = calculate_accuracy(y_pred_3, y_3)
        y_pred_4, y_4 = y_pred[:, 13:17], y[:, 13:17]
        acc_4 = calculate_accuracy(y_pred_4, y_4)
        y_pred_5, y_5 = y_pred[:, 17:23], y[:, 17:23]
        acc_5 = calculate_accuracy(y_pred_5, y_5)
        y_pred_6, y_6 = y_pred[:, 23:26], y[:, 23:26]
        acc_6 = calculate_accuracy(y_pred_6, y_6)
        acc = (acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6) / 6
        return acc