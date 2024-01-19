import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from FeedbackHHC import FeedbackHHC


class FeedbackHHCInterfaceGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("FeedbackHHC GUI")

        self.feedback_hhc = None
        self.file_path = None

        self.create_widgets()

    def create_widgets(self):
        # Load Data Button
        self.load_data_button = tk.Button(self.master, text="Load Data", command=self.load_data)
        self.load_data_button.pack(pady=10, side="top", anchor="center")

        # Preprocess Data Button
        self.preprocess_data_button = tk.Button(self.master, text="Preprocess Data", command=self.preprocess_data)
        self.preprocess_data_button.pack(pady=10, side="top", anchor="center")

        # Exploratory Analysis Button
        self.explore_data_button = tk.Button(self.master, text="Exploratory Analysis", command=self.explore_data)
        self.explore_data_button.pack(pady=10, side="top", anchor="center")

        # Select Attributes using PCA Button
        self.select_attributes_button = tk.Button(self.master, text="Select Attributes using PCA",
                                                  command=self.select_attributes_pca)
        self.select_attributes_button.pack(pady=10, side="top", anchor="center")

        # Train Random Forest Regressor Button
        self.train_regressor_button = tk.Button(self.master, text="Train Random Forest Regressor",
                                                command=self.train_random_forest_regressor)
        self.train_regressor_button.pack(pady=10, side="top", anchor="center")

        # Train Random Forest Classifier Button
        self.train_classifier_button = tk.Button(self.master, text="Train Random Forest Classifier",
                                                 command=self.train_random_forest_classifier)
        self.train_classifier_button.pack(pady=10, side="top", anchor="center")

        # Matplotlib Figure and Canvas
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(pady=10, side="top", anchor="center")

    def load_data(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            self.feedback_hhc = FeedbackHHC(self.file_path)
            messagebox.showinfo("Success", "Data loaded successfully!")

    def preprocess_data(self):
        if self.feedback_hhc:
            self.feedback_hhc.preprocess_data()
            messagebox.showinfo("Success", "Data preprocessed successfully!")
        else:
            messagebox.showinfo("Error", "An error occured at data preprocessing!")

    def explore_data(self):
        if self.feedback_hhc:
            self.explore_data_with_plot()

    def select_attributes_pca(self):
        if self.feedback_hhc:
            pca_results = self.feedback_hhc.select_attributes_pca()
            self.ax.clear()

            # Plot the explained variance ratio of each principal component
            self.ax.plot(np.arange(len(pca_results)), pca_results, marker='o')

            # Set the title and labels
            self.ax.set_title('PCA Results')
            self.ax.set_xlabel('Principal Component')
            self.ax.set_ylabel('Explained Variance Ratio')

            # Update the GUI
            self.canvas.draw()

    def train_random_forest_regressor(self):
        if self.feedback_hhc:
            self.feedback_hhc.train_random_forest_regressor()

    def train_random_forest_classifier(self):
        if self.feedback_hhc:
            self.feedback_hhc.train_random_forest_classifier()

    def update_plot(self):
        self.ax.clear()

        numeric_columns = self.feedback_hhc.data.select_dtypes(include=np.number).columns
        numeric_data = self.feedback_hhc.data[numeric_columns]

        print("The average for the attributes:\n", numeric_data.mean())
        print("\nThe median for the attributes:\n", numeric_data.median())

        for col in numeric_columns:
            self.ax.hist(numeric_data[col], bins=20, color='lightpink', edgecolor='black')
            self.ax.set_title(col)

        self.canvas.draw()

    def explore_data_with_plot(self):
        self.update_plot()




if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x800")
    app = FeedbackHHCInterfaceGUI(root)
    root.mainloop()
