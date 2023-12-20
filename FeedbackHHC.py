import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class FeedbackHHC:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocessdata(self):
        self.data.replace('-', 0, inplace=True)
        # scoate stringurile din coloanele numerice
        numeric_columns = self.data.select_dtypes(include='number').columns
        self.data[numeric_columns] = self.data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # nu avem duplicate, dar daca am fi avut, le-am fi eliminat cu:
        self.data.drop_duplicates(inplace=True)

        print("\nData after preprocessing:")
        print(self.data.head())

    def exploratory_analysis(self):
        numeric_data = self.data.select_dtypes(include=[np.number])
        mean_values = numeric_data.mean()
        median_values = numeric_data.median()

        print("Valorile medii:\n", mean_values)
        print("\nValorile mediane:\n", median_values)

        # vizualizarea datelor sub forma de histograma
        numeric_data.hist(figsize=(20, 20))
        plt.show()
