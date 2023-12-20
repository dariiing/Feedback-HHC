import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class FeedbackHHC:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    def type_number(self, column):
        for value in self.data[column]:
            if value != '-':
                try:
                    float(value)
                    return True
                except ValueError:
                    return False
            return True

    def preprocessdata(self):

        #eliminam duplicatele
        self.data.drop_duplicates(inplace=True)

        # stergem coloanele care au mai mult de 70% din randuri goale
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0.7 * len(self.data[col]):
                self.data.drop(col, axis=1, inplace=True)

        # inlocuim celulele goale cu valoarea 0
        for col in self.data.columns:
            if self.type_number(col):
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col].fillna(0, inplace=True)

        #inlocuim valorile numerice lipsa cu media
        for col in self.data.columns:
            if self.type_number(col):
                self.data[col].fillna(self.data[col].mean(), inplace=True)

        # scapam de outlieri


    def exploratory_analysis(self):
        numeric_columns = [col for col in self.data.columns if self.type_number(col)]
        numeric_data = self.data[numeric_columns]
        mean_values = numeric_data.mean()
        median_values = numeric_data.median()

        print("Valorile medii:\n", mean_values)
        print("\nValorile mediane:\n", median_values)

        #vizualizarea datelor sub forma de histograma
        numeric_data.hist(figsize=(20, 20))
        plt.show()