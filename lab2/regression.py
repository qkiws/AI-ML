import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class Regression:
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

        self.mse = None
        self.rmse = None
        self.mae = None

        self.errors = None

    def info(self):
        print('Датасет:')
        print(self.df.head())

        print('\nРазмер датасета:')
        print(self.df.shape)

    def split(self, test_size=0.2, random_state=42):
        self.X = self.df.drop('Close', axis=1)
        self.y = self.df['Close']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state
        )

        print('\nРазмер обучающей выборки:')
        print(self.X_train.shape)

        print('\nРазмер тестовой выборки:')
        print(self.X_test.shape)

    def init_model(self):
        self.split()

        self.model = LinearRegression()

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

        self.calc_errors()

    def calc_errors(self):
        self.mse = mean_squared_error(
                    self.y_test,
                    self.y_pred
                )
        
        self.rmse = self.mse ** 0.5

        self.mae = mean_absolute_error(
            self.y_test,
            self.y_pred
        )

        self.errors = self.y_pred - self.y_test.values

        print('\nMSE:', self.mse)
        print('RMSE:', self.rmse)
        print('MAE:', self.mae)

    def draw(self):
        plt.figure(figsize=(10, 5))

        plt.scatter(
            range(len(self.y_test)),
            self.y_test,
            label='Настоящие значения'
        )

        plt.plot(
            range(len(self.y_pred)),
            self.y_pred,
            label='Предсказанные значения'
        )

        plt.xlabel('Номер наблюдения')
        plt.ylabel('Close')
        plt.title('Линейная регрессия')
        plt.legend()

        plt.show()

        plt.figure(figsize=(10, 5))

        plt.scatter(
            range(len(self.errors)),
            self.errors
        )

        plt.axhline(0)

        plt.xlabel('Номер наблюдения')
        plt.ylabel('Ошибка')
        plt.title('Ошибки линейной регрессии')

        plt.show()


regression = Regression()

regression.init_model()
regression.draw()