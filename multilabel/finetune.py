import optuna
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils import *
from criterion import *

EPOCHS = 30


def finetune(model, train_iterator, valid_iterator, device):

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam"])
        }

        accuracy = train_and_evaluate(params, model, train_iterator, valid_iterator, device)

        return accuracy

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30)

    best_trial = study.best_trial

    torch.cuda.empty_cache()

    return best_trial.params


def train_and_evaluate(param, model, train_iterator, valid_iterator, device):

    FOUND_LR = param['learning_rate']

    params = [
        {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
        {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
        {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
        {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
        {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
        {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
        {'params': model.fc.parameters()}
    ]

    optimizer = getattr(optim, param['optimizer'])(params, lr=param['learning_rate'])

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
    best_valid_acc = 0

    for epoch in range(EPOCHS):

        train_loss, train_acc = train(model, train_iterator, optimizer, scheduler, criterion, accuracy, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, accuracy, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc

    return best_valid_acc