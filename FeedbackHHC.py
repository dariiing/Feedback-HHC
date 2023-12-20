import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class FeedbackHHC:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def numerical_types(self, col):
        index = self.data[col].first_valid_index()
        if isinstance(self.data[col][index], (int, float)):
            return True
        else:
            return False
    def preprocessdata(self):
        #inlocuim valorile numerice lipsa cu media
        for col in self.data.columns:
            if self.data[col].dtype == np.number:
                self.data[col].fillna(self.data[col].mean(), inplace=True)

        #stergem coloanele care au mai mult de 70% din randuri goale

        for col in self.data.columns:
                if self.data[col].isnull().sum() > 0.7 * len(self.data[col]):
                    self.data.drop(col, axis=1, inplace=True)

        # scapam de outlieri
        numeric_data = self.data.select_dtypes(include=[np.number])
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)
        outliers = numeric_data[((numeric_data < lower_bound) | (numeric_data > upper_bound)).any(axis=1)]
        self.data = self.data[~outliers].reset_index(drop=True)

    def exploratory_analysis(self):
        numeric_data = self.data.select_dtypes(include=[np.number])
        mean_values = numeric_data.mean()
        median_values = numeric_data.median()

        print("Valorile medii:\n", mean_values)
        print("\nValorile mediane:\n", median_values)

        #vizualizarea datelor sub forma de histograma
        numeric_data.hist(figsize=(20, 20))
        plt.show()