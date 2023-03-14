import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from dataset import FashionDataset
import os

BATCH_SIZE = 128
MODEL = 'resnet50'
OUTPUT_DIM = 26
CKPT = ''

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def output_labels(y_pred):

    y_prob = F.softmax(y_pred, dim = -1)
    top_pred = y_prob.argmax(1, keepdim=True)

    return top_pred


def get_predictions(model, iterator, device):

    model.eval()
    model.to(device)

    probs = []

    with torch.no_grad():

        for x, _ in iterator:

            x = x.to(device)

            y_pred = model(x)

            y_pred_1 = y_pred[:, :7]
            label_1 = output_labels(y_pred_1)
            y_pred_2 = y_pred[:, 7:10]
            label_2 = output_labels(y_pred_2)
            y_pred_3 = y_pred[:, 10:13]
            label_3 = output_labels(y_pred_3)
            y_pred_4 = y_pred[:, 13:17]
            label_4 = output_labels(y_pred_4)
            y_pred_5 = y_pred[:, 17:23]
            label_5 = output_labels(y_pred_5)
            y_pred_6 = y_pred[:, 23:26]
            label_6 = output_labels(y_pred_6)

            y_prob = torch.cat((label_1, label_2, label_3, label_4, label_5, label_6), 1)

            probs.append(y_prob.cpu())

    probs = torch.cat(probs, dim = 0)

    return probs


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_path = "../FashionDataset/split/test.txt"
    test_attr = "../FashionDataset/split/val_attr.txt"

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
        transforms.Resize(pretrained_size),
        transforms.CenterCrop(pretrained_size),
        transforms.ToTensor(),
        transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    test_data = FashionDataset(test_path, test_attr, test_transforms)

    test_iterator = data.DataLoader(test_data,
                                    shuffle=False,
                                    batch_size=BATCH_SIZE)

    checkpoint = torch.load(f'checkpoints/{CKPT}')

    if MODEL == 'resnet18':
        model = models.resnet18()
    elif MODEL == 'resnet34':
        model = models.resnet34()
    elif MODEL == 'resnet50':
        model = models.resnet50()
    elif MODEL == 'resnet152':
        model = models.resnet152()
    elif MODEL == 'resnet101':
        model = models.resnet101()

    IN_FEATURES = model.fc.in_features

    model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

    model.load_state_dict(checkpoint)

    pred_labels = get_predictions(model, test_iterator, device)

    pred_labels = pred_labels.numpy()

    ckpt_name = os.path.splitext(CKPT)[0]

    prediction = f'predictions/{ckpt_name}-prediction.txt'

    file_write_obj = open(prediction, 'a')
    for pred_label in pred_labels:
        input = f'{pred_label[0]} {pred_label[1]} {pred_label[2]} {pred_label[3]} {pred_label[4]} {pred_label[5]}'
        file_write_obj.writelines(input)
        file_write_obj.write('\n')
    file_write_obj.close()
