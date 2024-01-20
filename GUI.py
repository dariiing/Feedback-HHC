import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from FeedbackHHC import FeedbackHHC


class FeedbackHHCInterfaceGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("FeedbackHHC")

        self.pca_plot = 0
        self.style = ttk.Style()
        self.style.theme_use("clam")

        main_frame = tk.Frame(self.master)
        main_frame.pack(fill="both", expand=True)

        button_frame = tk.Frame(main_frame, width=200, bg="lightgray")
        button_frame.pack(side="left", fill="y", padx=10)

        style = ttk.Style()
        style.configure("TButton", padding=(10, 5, 10, 5), font=('Helvetica', 10))

        self.load_data_button = ttk.Button(button_frame, text="Load Data", command=self.load_data)
        self.load_data_button.pack(pady=10, anchor="w", ipadx=5)

        self.preprocess_data_button = ttk.Button(button_frame, text="Preprocess Data", command=self.preprocess_data)
        self.preprocess_data_button.pack(pady=10, anchor="w", ipadx=5)

        self.explore_data_button = ttk.Button(button_frame, text="Exploratory Analysis",
                                              command=self.explore_data_with_plot)
        self.explore_data_button.pack(pady=10, anchor="w", ipadx=5)

        self.select_attributes_button = ttk.Button(button_frame, text="Select Attributes using PCA",
                                                   command=self.select_attributes_pca)
        self.select_attributes_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_regressor_button = ttk.Button(button_frame, text="Train Random Forest Regressor",
                                                 command=self.train_random_forest_regressor)
        self.train_regressor_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_classifier_button = ttk.Button(button_frame, text="Train Random Forest Classifier",
                                                  command=self.train_random_forest_classifier)
        self.train_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_svm_classifier_button = ttk.Button(button_frame, text="Train SVM Classifier",
                                                      command=self.train_svm_classifier)
        self.train_svm_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_decision_tree_classifier_button = ttk.Button(button_frame, text="Train Decision Tree Classifier",
                                                                command=self.train_decision_tree_classifier)
        self.train_decision_tree_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_neural_network_classifier_button = ttk.Button(button_frame, text="Train Neural Network Classifier",
                                                                 command=self.train_neural_network_classifier)
        self.train_neural_network_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.compare_models_performance_button = ttk.Button(button_frame, text="Compare Models Performance",
                                                            command=self.compare_models_performance)
        self.compare_models_performance_button.pack(pady=10, anchor="w", ipadx=5)

        self.compare_models_performance_multiclass_button = ttk.Button(button_frame,
                                                                       text="Compare Models Performance (Multi-Class)",
                                                                       command=self.compare_models_performance_multiclass)
        self.compare_models_performance_multiclass_button.pack(pady=10, anchor="w", ipadx=5)

        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side="right", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(pady=10, side="top", anchor="center", fill="both", expand=True)

        self.feedback_hhc = None
        self.file_path = None

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
            messagebox.showinfo("Error", "An error occurred at data preprocessing!")

    def explore_data_with_plot(self):
        if self.feedback_hhc:
            numeric_columns = self.feedback_hhc.data.select_dtypes(include=np.number).columns
            numeric_data = self.feedback_hhc.data[numeric_columns]

            print("The average for the attributes:\n", numeric_data.mean())
            print("\nThe median for the attributes:\n", numeric_data.median())

            self.pca_plot = 0

            self.update_plot(numeric_columns, numeric_data)

    def select_attributes_pca(self):
        if self.feedback_hhc:
            pca_results = self.feedback_hhc.select_attributes_pca()

            self.ax.clear()

            self.ax.plot(np.arange(len(pca_results)), pca_results, marker='o')
            self.ax.set_title('PCA Results')
            self.ax.set_xlabel('Principal Component')
            self.ax.set_ylabel('Explained Variance Ratio')

            self.canvas.draw()
            self.pca_plot = 1

    def update_plot(self, numeric_columns, numeric_data, index=0):
        if index < len(numeric_columns) and self.pca_plot == 0:
            col = numeric_columns[index]

            self.ax.clear()

            self.ax.hist(numeric_data[col], bins=20, color='lightpink', edgecolor='black')
            self.ax.set_title(col)

            self.canvas.draw()

            self.master.after(2000, lambda: self.update_plot(numeric_columns, numeric_data, index + 1))
        else:
            self.master.update()

    def train_random_forest_regressor(self):
        if self.feedback_hhc:
            self.feedback_hhc.train_random_forest_regressor()

    def train_random_forest_classifier(self):
        if self.feedback_hhc:
            self.feedback_hhc.train_random_forest_classifier()

    def train_svm_classifier(self):
        if self.feedback_hhc:
            self.feedback_hhc.train_svm_classifier()

    def train_decision_tree_classifier(self):
        if self.feedback_hhc:
            self.feedback_hhc.train_decision_tree_classifier()

    def train_neural_network_classifier(self):
        if self.feedback_hhc:
            self.feedback_hhc.train_neural_network_classifier()

    def compare_models_performance(self):
        if self.feedback_hhc:
            self.feedback_hhc.compare_models_performance()

    def compare_models_performance_multiclass(self):
        if self.feedback_hhc:
            self.feedback_hhc.compare_models_performance_multiclass()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = FeedbackHHCInterfaceGUI(root)
    root.mainloop()
