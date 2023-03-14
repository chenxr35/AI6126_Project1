import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import torchvision.transforms as transforms
import random
import numpy as np
import time
import os
import datetime
import copy
from dataset import FashionDataset
from utils import *
from criterion import *
from finetune import finetune

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

BATCH_SIZE = 64
OUTPUT_DIM = 26
EPOCHS = 100
MODEL = 'resnet152'
FINETUNE = True


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    now = datetime.datetime.now()
    log = f'logs/finetune{FINETUNE}-{MODEL}-{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}.txt'
    checkpoint = f'checkpoints/finetune{FINETUNE}-{MODEL}-{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}.pth'

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    train_path = "../FashionDataset/split/train.txt"
    train_attr = "../FashionDataset/split/train_attr.txt"
    val_path = "../FashionDataset/split/val.txt"
    val_attr = "../FashionDataset/split/val_attr.txt"

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize(pretrained_size),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(pretrained_size, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(pretrained_means, pretrained_stds)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(pretrained_size),
        transforms.CenterCrop(pretrained_size),
        transforms.ToTensor(),
        transforms.Normalize(pretrained_means, pretrained_stds)
    ])

    train_data = FashionDataset(train_path, train_attr, train_transforms)
    val_data = FashionDataset(val_path, val_attr, val_transforms)
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(val_data)}')

    train_iterator = data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=BATCH_SIZE)

    valid_iterator = data.DataLoader(val_data,
                                     batch_size=BATCH_SIZE)

    if MODEL == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
    elif MODEL == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1')
    elif MODEL == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif MODEL == 'resnet101':
        model = models.resnet101(weights='IMAGENET1K_V1')
    elif MODEL == 'resnet152':
        model = models.resnet152(weights='IMAGENET1K_V1')

    IN_FEATURES = model.fc.in_features

    model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

    num_of_params = f'The model has {count_parameters(model):,} trainable parameters'
    print(num_of_params)
    file_write_obj = open(log, 'a')
    file_write_obj.writelines(num_of_params)
    file_write_obj.write('\n')
    file_write_obj.close()

    finetune_param = finetune(copy.deepcopy(model), train_iterator, valid_iterator, device)

    FOUND_LR = finetune_param['learning_rate']
    FOUND_OPTIM = finetune_param['optimizer']

    finetune_config = f'optimizer: {FOUND_OPTIM} | learning rate: {FOUND_LR}'
    print(finetune_config)
    file_write_obj = open(log, 'a')
    file_write_obj.writelines(finetune_config)
    file_write_obj.write('\n')
    file_write_obj.close()

    params = [
        {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
        {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
        {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
        {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
        {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
        {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
        {'params': model.fc.parameters()}
    ]

    optimizer = getattr(optim, finetune_param['optimizer'])(params, lr=FOUND_LR)

    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

    MAX_LRS = [p['lr'] for p in optimizer.param_groups]

    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr=MAX_LRS,
                                        total_steps=TOTAL_STEPS)

    model = model.to(device)

    criterion = FashionLoss()
    criterion.to(device)
    accuracy = FashionAccuracy()
    accuracy.to(device)

    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):

        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_iterator, optimizer, scheduler, criterion, accuracy, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, accuracy, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        epoch_log = f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s'
        train_log = f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%'
        valid_log = f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%'
        line = f'------------------------------------------------------------------'

        print(epoch_log)
        print(train_log)
        print(valid_log)

        loglist = [epoch_log, train_log, valid_log, line]

        file_write_obj = open(log, 'a')
        for var in loglist:
            file_write_obj.writelines(var)
            file_write_obj.write('\n')
        file_write_obj.close()







