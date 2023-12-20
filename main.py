from FeedbackHHC import FeedbackHHC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    data = FeedbackHHC(r'HH_Provider_Oct2023.csv')
    print("Data before preprocessing:")
    print(data.data)

    data.preprocessdata()
    data.exploratory_analysis()


if __name__ == "__main__":
    main()
