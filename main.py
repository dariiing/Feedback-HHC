from FeedbackHHC import FeedbackHHC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    data = FeedbackHHC(r'HH_Provider_Oct2023.csv')
    ok=0
    for col in data.data.columns:
        if data.type_number(col):
            ok+=1
            print(col)
    print(ok)
    data.preprocessdata()
    data.exploratory_analysis()

    data.select_attributes()


if __name__ == "__main__":
    main()