import numpy as np
import seaborn as sns
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


labels = ['LGG', 'HGG', 'Healthy']

c = ([[399  , 1767 , 138],
       [306   , 7033  , 149],
       [61 , 148 , 6991]])

# c = ([[854, 417, 1],
#        [269, 3862, 3],
#        [12, 30, 3933]])

c=np.asarray(c)
group_names = ['LGG', 'HGG', 'Healthy']

k = c / c.astype(np.float).sum(axis=1)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


alpha= ['LGG', 'HGG', 'Healthy']

plot_confusion_matrix(c, classes=alpha, normalize=True)

