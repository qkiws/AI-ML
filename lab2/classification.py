
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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

        self.accuracy = None
        self.cm = None
        self.precision = None
        self.recall = None
        self.f1 = None

        self.improved_model = None
        self.improved_y_pred = None

        self.improved_accuracy = None
        self.improved_cm = None
        self.improved_precision = None
        self.improved_recall = None
        self.improved_f1 = None

    def split(self, test_size=0.2, random_state=42):
        self.df['Class'] = (
            self.df['Close'] > self.df['Open']
        ).astype(int)

        self.X = self.df[
            ['High', 'Low', 'Volume', 'Year', 'Month', 'Day']
        ]

        self.y = self.df['Class']

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )

        print('\nРазмер обучающей выборки:')
        print(self.X_train.shape)

        print('\nРазмер тестовой выборки:')
        print(self.X_test.shape)

    def init_model(self):
        self.split()

        self.model = LogisticRegression()

        self.model.fit(
            self.X_train,
            self.y_train
        )

        self.y_pred = self.model.predict(
            self.X_test
        )

        print('\nПервые предсказания:')
        print(self.y_pred[:10])

        print('\nНастоящие значения:')
        print(self.y_test.values[:10])

        self.accuracy = accuracy_score(
            self.y_test,
            self.y_pred
        )

        self.cm = confusion_matrix(
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

        print('\nAccuracy:', self.accuracy)

        print('\nConfusion Matrix:')
        print(self.cm)

        print('\nPrecision:', self.precision)
        print('Recall:', self.recall)
        print('F1:', self.f1)

    def improve_model(self):
        self.improved_model = LogisticRegression(
            class_weight='balanced'
        )

        self.improved_model.fit(
            self.X_train,
            self.y_train
        )

        self.improved_y_pred = self.improved_model.predict(
            self.X_test
        )

        self.improved_accuracy = accuracy_score(
            self.y_test,
            self.improved_y_pred
        )

        self.improved_cm = confusion_matrix(
            self.y_test,
            self.improved_y_pred
        )

        self.improved_precision = precision_score(
            self.y_test,
            self.improved_y_pred
        )

        self.improved_recall = recall_score(
            self.y_test,
            self.improved_y_pred
        )

        self.improved_f1 = f1_score(
            self.y_test,
            self.improved_y_pred
        )

        print('\n==========================================')
        print('УЛУЧШЕННАЯ КЛАССИФИКАЦИЯ')
        print('==========================================')

        print('\nAccuracy:', self.improved_accuracy)

        print('\nConfusion Matrix:')
        print(self.improved_cm)

        print('\nPrecision:', self.improved_precision)
        print('Recall:', self.improved_recall)
        print('F1:', self.improved_f1)

    def draw(self):
        plt.figure(figsize=(4, 3))

        sns.heatmap(
            self.cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )

        plt.title('Confusion matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.show()

    def draw_improved(self):
        plt.figure(figsize=(4, 3))

        sns.heatmap(
            self.improved_cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )

        plt.title('Improved confusion matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.show()

    def draw_sigmoid(self):
        x = np.linspace(-10, 10, 100)

        y = 1 / (1 + np.exp(-x))

        plt.figure(figsize=(8, 5))

        plt.plot(x, y)

        plt.axhline(0.5)
        plt.axvline(0)

        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.title('Sigmoid function')

        plt.show()

classification = Classification()

classification.init_model()
classification.improve_model()

classification.draw()
classification.draw_improved()
classification.draw_sigmoid()

