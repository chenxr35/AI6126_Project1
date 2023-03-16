import matplotlib.pyplot as plt


train_loss = [0.643, 0.520, 0.434, 0.372, 0.326, 0.294, 0.270, 0.248, 0.230, 0.211, 0.195, 0.177, 0.161, 0.144, 0.132]
val_loss   = [0.576, 0.472, 0.394, 0.341, 0.306, 0.280, 0.260, 0.245, 0.232, 0.224, 0.216, 0.213, 0.214, 0.211, 0.217]
train_acc  = [0.4193,0.6369,0.6700,0.6913,0.7168,0.7394,0.7599,0.7835,0.8013,0.8224,0.8384,0.8533,0.8705,0.8860,0.8989]
val_acc    = [0.5893,0.6586,0.6849,0.7065,0.7305,0.7518,0.7697,0.7835,0.7949,0.8010,0.8090,0.8144,0.8110,0.8194,0.8145]

if __name__ == '__main__':

    epoch = [i+1 for i in range(len(train_loss))]

    plt.plot(epoch, train_loss, label='training loss',   color='m', marker='o', linestyle='dashed')
    plt.plot(epoch, val_loss,   label='validation loss', color='c', marker='*', linestyle='dashed')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig('loss_curve.png')

    plt.close()

    plt.plot(epoch, train_acc, label='training accuracy', color='r', marker='o', linestyle='dashed')
    plt.plot(epoch, val_acc,   label='validation accuracy', color='b', marker='*', linestyle='dashed')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.savefig('acc_curve.png')

