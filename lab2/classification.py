
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
        self.df['Class'] = (self.df["Close"] > 0).astype(int)

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

    def drawErrors(self):
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

    def draw_classes(self):
        counts = self.y.value_counts().sort_index()

        plt.figure(figsize=(5, 4))
        plt.bar(['Class 0', 'Class 1'], counts.values)

        plt.title('Распределение классов')
        plt.xlabel('Класс')
        plt.ylabel('Количество объектов')

        plt.show()

    def draw(self):
        self.drawErrors()
        self.draw_classes()

classification = Classification()

classification.init_model()

classification.draw()


