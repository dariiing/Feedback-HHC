from FeedbackHHC import FeedbackHHC


def main():
    data = FeedbackHHC(r'HH_Provider_Oct2023.csv')
    print("Data before preprocessing:")
    print(data.data)

    data.preprocess_data()
    # data.exploratory_analysis()


if __name__ == "__main__":
    main()
