import matplotlib.pyplot as plt
import os


def show_train_history(history, train, validation, store_path=r''):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(os.path.join(store_path, 'tendency.png'))

# load the result

# show_train_history(history, 'acc', 'val_acc')
# show_train_history(history, 'loss', 'val_loss')
