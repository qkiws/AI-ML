import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DatasetProcessor:
    def __init__(self, file_name='df.csv'):
        self.file_path = os.path.join(os.getcwd(), file_name)
        self.done_path = os.path.join(os.getcwd(), 'done.csv')
        self.df = None

    def load_from_kaggle(self):
        self.df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "hassanjameelahmed/thob-al-aseel-company-dataset",
            "Thob Al Aseel Company.csv"
        )

    def save_to_csv(self, path=None):
        if self.df is None:
            print("Ошибка: Нет данных для сохранения.")
            return

        self.df.to_csv(self.done_path, index=False)
        print(f'Сохранено в: {self.done_path}')

        

    def load_from_csv(self):
        self.df = pd.read_csv(self.file_path)

    def console_log(self, n: int = 3):
        if self.df is not None:
            print(f"Первые {n} строк датасета:")
            print(self.df.head(n))
        else:
            print("Датасет пуст.")

    def count_null(self):
        if self.df is not None:
            return self.df.isnull().sum()
        return None

    def fill_missing_values(self):
        if self.df is None:
            return

        print("\nЗаполнение пропущенных значений")
        categorical_cols = self.df.select_dtypes(exclude='number').columns
        print('Все доступные колонки', self.df.columns.to_list())
        for col in categorical_cols:
            col_mode = self.df[col].mode()[0]
            self.df[col] = self.df[col].fillna(col_mode)

        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            col_median = self.df[col].median()
            self.df[col] = self.df[col].fillna(col_median)
        print("Пропуски успешно заполнены!")

    def normalize(self):
        if self.df is None:
            return

        numeric_cols = self.df.select_dtypes(include='number').columns

        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        print('Нормализация готова')

    def date(self):
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])

            self.df['Year'] = self.df['Date'].dt.year
            self.df['Month'] = self.df['Date'].dt.month
            self.df['Day'] = self.df['Date'].dt.day

            self.df = self.df.drop(columns=['Date'])

    def OHE(self):
        if self.df is None:
            return

        ohe_columns = self.df.select_dtypes(exclude='number').columns
        self.df = pd.get_dummies(self.df, columns=ohe_columns, drop_first=True)
        print('OHE выполнено')

    def setup(self):
        if os.path.exists(self.file_path) and os.stat(self.file_path).st_size != 0:
            print('Инициализировано, берем из csv')
            self.load_from_csv()
        else:
            print('Пусто, загружаем в csv')
            self.load_from_kaggle()
            self.save_to_csv()


def run():
    runner = DatasetProcessor()
    runner.setup()
    runner.console_log()
    print("\n\nПропуски ДО\n\n", runner.count_null())
    runner.fill_missing_values()
    print("\n\nПропуски ПОСЛЕ\n\n", runner.count_null())
    runner.date()
    print('Нормализация...')
    runner.normalize()
    runner.OHE()
    runner.save_to_csv()


if __name__ == '__main__':
    run()
