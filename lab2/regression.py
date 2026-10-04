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

    def check_errors(self):
        print('\n\n\nОтдельный блок для оценки переобучения')
        print('\nКоэффы:', self.model.coef_)
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)

        train_rmse = pow(train_mse, 1/2)
        test_rmse = pow(test_mse, 1/2)

        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)

        print("Train MSE:", train_mse)
        print("Test MSE:", test_mse)

        print("Train RMSE:", train_rmse)
        print("Test RMSE:", test_rmse)

        print("Train MAE:", train_mae)
        print("Test MAE:", test_mae)

    def draw_MSE(self):
        
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        train_mse = mean_squared_error(self.y_train, train_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)

        plt.figure(figsize=(6, 4))

        plt.bar(
            ['Train', 'Test'],
            [train_mse, test_mse]
        )

        plt.title('MSE на обучающей и тестовой выборках')
        plt.ylabel('MSE')

        plt.show()

        errors = self.y_test - self.y_pred

    def draw_errors(self):
        plt.figure(figsize=(10, 4))
        plt.scatter(
            range(len(self.errors)),
            self.errors,
            s=15
        )

        plt.axhline(0)

        plt.title('Ошибки модели на тестовой выборке')
        plt.xlabel('Номер объекта')
        plt.ylabel('Ошибка')

        plt.grid()
        plt.show()
    def draw_predict(self):
        plt.figure(figsize=(10, 7))

        plt.scatter(
            self.y_test,
            self.y_pred,
            s=20,
            label="Предсказания"
        )

        min_value = min(self.y_test.min(), self.y_pred.min())
        max_value = max(self.y_test.max(), self.y_pred.max())

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            label="Идеальное предсказание"
        )

        plt.xlabel("Настоящие значения")
        plt.ylabel("Предсказанные значения")
        plt.title("Настоящие значения и предсказания")

        plt.legend()
        plt.grid()

        plt.show()

    def draw(self):
        self.draw_predict()
        self.draw_errors()
        self.draw_MSE()


regression = Regression()

regression.init_model()

regression.check_errors()

regression.draw()