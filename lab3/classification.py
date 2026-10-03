import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

class Classification:
    def __init__(self):
        self.file_path = os.path.join(os.getcwd(), 'done.csv')

        self.df = pd.read_csv(self.file_path)

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None
        self.y_pred = None
        self.y_proba = None

        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None

        self.fpr = None
        self.tpr = None
        self.roc_auc = None

    def prepare_data(self):
        self.df['Target'] = (self.df['Close'] > 0).astype(int)

        self.X = self.df.drop(
            ['Close', 'Target'],
            axis=1
        )

        self.y = self.df['Target']

    def split(self, test_size=0.2, random_state=42):
        self.prepare_data()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )

        print('\nРазмер обучающей выборки классификации:')
        print(self.X_train.shape)

        print('\nРазмер тестовой выборки классификации:')
        print(self.X_test.shape)

    def init_model(self):
        self.split()

        self.model = DecisionTreeClassifier(
            max_depth=4,
            random_state=42,
        )

        self.model.fit(
            self.X_train,
            self.y_train
        )

        self.y_pred = self.model.predict(
            self.X_test
        )

        self.y_proba = self.model.predict_proba(
            self.X_test
        )[:, 1]

        self.calc_metrics()

    def calc_metrics(self):
        self.accuracy = accuracy_score(
            self.y_test,
            self.y_pred
        )

        self.precision = precision_score(
            self.y_test,
            self.y_pred
        )

        self.recall = recall_score(
            self.y_test,
            self.y_pred
        )

        self.f1 = f1_score(
            self.y_test,
            self.y_pred
        )

        print('\nРезультаты классификации:')
        print('Accuracy:', self.accuracy)
        print('Precision:', self.precision)
        print('Recall:', self.recall)
        print('F1:', self.f1)

        print('\nМатрица ошибок:')
        print(confusion_matrix(
            self.y_test,
            self.y_pred
        ))

    def draw_roc(self):
        self.fpr, self.tpr, thresholds = roc_curve(
            self.y_test,
            self.y_proba
        )

        self.roc_auc = auc(
            self.fpr,
            self.tpr
        )

        print('\nROC-AUC:', self.roc_auc)

        plt.figure(figsize=(10, 6))

        plt.plot(
            self.fpr,
            self.tpr,
            color='blue',
            linewidth=3,
            label='ROC-кривая (AUC = {:.3f})'.format(self.roc_auc)
        )

        plt.plot(
            [0, 1],
            [0, 1],
            color='red',
            linestyle='--',
            linewidth=2,
            label='Случайная классификация'
        )

        plt.fill_between(
            self.fpr,
            self.tpr,
            alpha=0.15,
            color='blue'
        )

        plt.xlim([0, 1])
        plt.ylim([0, 1.05])

        plt.xlabel('FPR — доля ложноположительных')
        plt.ylabel('TPR — доля истинноположительных')

        plt.title('ROC-кривая дерева решений')

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.show()
    
    def draw_tree(self):
        plt.figure(figsize=(20, 10))

        plot_tree(
            self.model,
            feature_names=self.X.columns,
            class_names=['0', '1'],
            filled=True
        )

        plt.title('Дерево решений для классификации')

        plt.show()

    def draw(self):
        self.draw_roc()
        self.draw_tree()



print('\n\n======================================')
print('КЛАССИФИКАЦИЯ')
print('======================================')

classification = Classification()

classification.init_model()
classification.draw_roc()
classification.draw_tree()