import torch.nn as nn
from utils import calculate_accuracy


class FashionAccuracy(nn.Module):

    def __init__(self):
        super(FashionAccuracy, self).__init__()

    def forward(self, y_pred, y):
        y_pred_1, y_1 = y_pred[:, :7], y[:, 0]
        acc_1 = calculate_accuracy(y_pred_1, y_1)
        y_pred_2, y_2 = y_pred[:, 7:10], y[:, 1]
        acc_2 = calculate_accuracy(y_pred_2, y_2)
        y_pred_3, y_3 = y_pred[:, 10:13], y[:, 2]
        acc_3 = calculate_accuracy(y_pred_3, y_3)
        y_pred_4, y_4 = y_pred[:, 13:17], y[:, 3]
        acc_4 = calculate_accuracy(y_pred_4, y_4)
        y_pred_5, y_5 = y_pred[:, 17:23], y[:, 4]
        acc_5 = calculate_accuracy(y_pred_5, y_5)
        y_pred_6, y_6 = y_pred[:, 23:26], y[:, 5]
        acc_6 = calculate_accuracy(y_pred_6, y_6)
        acc = (acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6) / 6
        return acc


class FashionLoss(nn.Module):

    def __init__(self):
        super(FashionLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()

    def forward(self, y_pred, y):
        y_pred_1, y_1 = y_pred[:,:7], y[:,0]
        loss_1 = self.CE(y_pred_1, y_1)
        y_pred_2, y_2 = y_pred[:,7:10], y[:,1]
        loss_2 = self.CE(y_pred_2, y_2)
        y_pred_3, y_3 = y_pred[:,10:13], y[:,2]
        loss_3 = self.CE(y_pred_3, y_3)
        y_pred_4, y_4 = y_pred[:,13:17], y[:,3]
        loss_4 = self.CE(y_pred_4, y_4)
        y_pred_5, y_5 = y_pred[:,17:23], y[:,4]
        loss_5 = self.CE(y_pred_5, y_5)
        y_pred_6, y_6 = y_pred[:,23:26], y[:,5]
        loss_6 = self.CE(y_pred_6, y_6)
        loss = (loss_1+loss_2+loss_3+loss_4+loss_5+loss_6)/6
        return loss