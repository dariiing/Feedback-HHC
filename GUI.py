import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import roc_curve, auc, roc_auc_score

from FeedbackHHC import FeedbackHHC


class FeedbackHHCInterfaceGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("FeedbackHHC")

        self.pca_plot = 0
        self.rf_plot = 0
        self.rfc_plot = 0
        self.dtc_plot = 0
        self.dtr_plot = 0
        self.dtcm_plot = 0
        self.svmc_plot = 0
        self.svmm_plot = 0
        self.nn_plot = 0
        self.nnc_plot = 0
        self.cmp_plot = 0
        self.cmpm_plot = 0
        self.svmm_plot = 0
        self.style = ttk.Style()
        self.style.theme_use("clam")

        main_frame = tk.Frame(self.master)
        main_frame.pack(fill="both", expand=True)

        button_frame = tk.Frame(main_frame, width=200, bg="lightgray")
        button_frame.pack(side="left", fill="y", padx=5)

        style = ttk.Style()
        style.configure("TButton", padding=(10, 5, 10, 5), font=('Helvetica', 10))

        self.load_data_button = ttk.Button(button_frame, text="Load Data", command=self.load_data, width=25)
        self.load_data_button.pack(pady=10, anchor="w", ipadx=5)

        self.preprocess_data_button = ttk.Button(button_frame, text="Preprocess Data", command=self.preprocess_data,
                                                 width=25)
        self.preprocess_data_button.pack(pady=10, anchor="w", ipadx=5)

        self.explore_data_button = ttk.Button(button_frame, text="Exploratory Analysis",
                                              command=self.explore_data_with_plot, width=25)
        self.explore_data_button.pack(pady=10, anchor="w", ipadx=5)

        self.select_attributes_button = ttk.Button(button_frame, text="Select Attributes using PCA",
                                                   command=self.select_attributes_pca, width=25)
        self.select_attributes_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_regressor_button = ttk.Button(button_frame, text="Train Random Forest Regressor",
                                                 command=self.train_random_forest_regressor, width=25)
        self.train_regressor_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_classifier_button = ttk.Button(button_frame, text="ROC(Random Forest Regressor)",
                                                  command=self.train_random_forest_classifier_multiclass, width=25)
        self.train_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_classifier_button = ttk.Button(button_frame, text="ROC(Random Forest Classifier)",
                                                  command=self.train_random_forest_classifier, width=25)
        self.train_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_regressor_button = ttk.Button(button_frame, text="Train SVM Regressor",
                                                 command=self.train_svm_regressor, width=25)
        self.train_regressor_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_svm_classifier_button = ttk.Button(button_frame, text="ROC(SVM Classifier)",
                                                      command=self.train_svm_classifier, width=25)
        self.train_svm_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_svm_classifier_multi_class_button = ttk.Button(button_frame, text="ROC(SVM Multi-class)",
                                                                  command=self.train_svm_classifier_multi_class,
                                                                  width=25)
        self.train_svm_classifier_multi_class_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_regressor_button = ttk.Button(button_frame, text="Train Decision Tree Regressor",
                                                 command=self.train_decision_tree_regressor, width=25)
        self.train_regressor_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_decision_tree_classifier_button = ttk.Button(button_frame, text="ROC(Decision Tree Regressor)",
                                                                command=self.train_decision_tree_classifier_multiclass,
                                                                width=25)
        self.train_decision_tree_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_decision_tree_classifier_button = ttk.Button(button_frame, text="ROC(Decision Tree Classifier)",
                                                                command=self.train_decision_tree_classifier, width=25)
        self.train_decision_tree_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_neural_network_button = ttk.Button(button_frame, text="Train Neural Network",
                                                      command=self.train_neural_network, width=25)
        self.train_neural_network_button.pack(pady=10, anchor="w", ipadx=5)

        self.train_neural_network_classifier_button = ttk.Button(button_frame, text="ROC(Neural Network Classifier)",
                                                                 command=self.train_neural_network_classifier, width=25)
        self.train_neural_network_classifier_button.pack(pady=10, anchor="w", ipadx=5)

        self.compare_models_performance_button = ttk.Button(button_frame, text="Compare Models Performance",
                                                            command=self.compare_models_performance, width=25)
        self.compare_models_performance_button.pack(pady=10, anchor="w", ipadx=5)

        self.compare_models_performance_multiclass_button = ttk.Button(button_frame,
                                                                       text="Compare Models Performance (Multi-Class)",
                                                                       command=self.compare_models_performance_multiclass,
                                                                       width=25)
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
        else:
            messagebox.showinfo("Error", "An error occurred at data loading!")

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
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0

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
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0

    def update_plot(self, numeric_columns, numeric_data, index=0):
        if index < len(
                numeric_columns) and self.pca_plot == 0 and self.rf_plot == 0 and self.rfc_plot == 0 and self.dtc_plot == 0 and self.dtr_plot == 0 and self.dtcm_plot == 0 and self.svmc_plot == 0 and self.svmm_plot == 0 and self.nn_plot == 0 and self.nnc_plot == 0:
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
            self.rf_plot = 1
            self.pca_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0
            results = self.feedback_hhc.train_random_forest_regressor()
            y_test = results['y_test']
            y_pred_rounded = results['y_pred_rounded']
            mae = results['mae']
            r2 = results['r2']

            self.ax.clear()

            self.ax.scatter(y_test, y_pred_rounded, color='blue')
            self.ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red',
                         linewidth=2)
            self.ax.set_xlabel('Actual Values')
            self.ax.set_ylabel('Predicted Values (Rounded)')
            self.ax.set_title(
                f'Actual vs Predicted Values - Random Forest Regressor (Rounded)\nMAE: {mae:.2f}, R2: {r2:.2f}')

            self.canvas.draw()

    def train_random_forest_classifier(self):
        if self.feedback_hhc:
            self.rfc_plot = 1
            self.rf_plot = 1
            self.pca_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0
            results = self.feedback_hhc.train_random_forest_classifier()
            y_test = results['y_test']
            y_pred_prob = results['y_pred_prob']
            accuracy = results['accuracy']
            conf_matrix = results['conf_matrix']

            fpr_0, tpr_0, _ = roc_curve(y_test, y_pred_prob[:, 0])
            roc_auc_0 = auc(fpr_0, tpr_0)

            fpr_1, tpr_1, _ = roc_curve(y_test, y_pred_prob[:, 1])
            roc_auc_1 = auc(fpr_1, tpr_1)

            self.ax.clear()
            self.ax.plot(fpr_0, tpr_0, color='darkorange', lw=2, label='AUC(Class 0) (area = %0.2f)' % roc_auc_0)
            self.ax.plot(fpr_1, tpr_1, color='green', lw=2, label='AUC(Class 1) (area = %0.2f)' % roc_auc_1)
            self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.ax.set_xlabel('False Positive Rate')
            self.ax.set_ylabel('True Positive Rate')
            self.ax.set_title(
                f'Receiver operating characteristic \nAccuracy: {accuracy:.2f} , Confusion Matrix: \n{conf_matrix}')
            self.ax.legend(loc="lower right")
            self.canvas.draw()



    def train_random_forest_classifier_multiclass(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.train_random_forest_classifier_multiclass()
            y = results['y']
            y_test = results['y_test']
            y_pred_prob = results['y_pred_prob']

            self.ax.clear()
            for i in range(len(np.unique(y))):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                self.ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

            self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.ax.set_xlabel('False Positive Rate')
            self.ax.set_ylabel('True Positive Rate')
            self.ax.set_title(f'Receiver operating characteristic')
            self.ax.legend(loc="lower right")
            self.canvas.draw()

            classes = np.unique(y)
            for i in range(len(classes)):
                class_auc = roc_auc_score(y_test[:, i], y_pred_prob[:, i])
                print(f"AUC for Class '{classes[i]}': {class_auc:.2%}")

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 1
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0

    def train_svm_regressor(self):
        if self.feedback_hhc:
            self.rf_plot = 0
            self.pca_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 1
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0
            results = self.feedback_hhc.train_svm_regressor()
            y_test = results['y_test']
            y_pred_rounded = results['y_pred_rounded']
            mae = results['mae']
            r2 = results['r2']

            self.ax.clear()

            self.ax.scatter(y_test, y_pred_rounded, color='blue')
            self.ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red',
                         linewidth=2)
            self.ax.set_xlabel('Actual Values')
            self.ax.set_ylabel('Predicted Values (Rounded)')
            self.ax.set_title(
                f'Actual vs Predicted Values - SVM Regressor (Rounded)\nMAE: {mae:.2f}, R2: {r2:.2f}')

            self.canvas.draw()

    def train_svm_classifier(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.train_svm_classifier()
            y_test = results['y_test']
            y_pred_prob = results['y_pred_prob']
            accuracy = results['accuracy']
            conf_matrix = results['conf_matrix']

            fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
            roc_auc = auc(fpr, tpr)

            self.ax.clear()
            self.ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.ax.set_xlabel('False Positive Rate')
            self.ax.set_ylabel('True Positive Rate')
            self.ax.set_title(
                f'Receiver operating characteristic -SVM Classifier \nAccuracy: {accuracy:.2f} , Confusion Matrix: \n{conf_matrix}')
            self.ax.legend(loc='lower right')
            self.canvas.draw()

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 1
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0

    def train_decision_tree_regressor(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.train_decision_tree_regressor()
            y_test = results['y_test']
            y_pred = results['y_pred']
            mse = results['mse']

            self.ax.clear()
            self.ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
            self.ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
            self.ax.set_xlabel('Actual')
            self.ax.set_ylabel('Predicted')
            self.ax.set_title(f'Decision Tree Regressor \nMean Squared Error: {mse:.2f}')
            self.canvas.draw()

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 1
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0

    def train_svm_classifier_multi_class(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.train_svm_classifier_multiclass()
            y = results['y']
            y_test = results['y_test']
            y_pred_prob = results['y_pred_prob']

            self.ax.clear()
            for i in range(len(np.unique(y))):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                self.ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

            self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.ax.set_xlabel('False Positive Rate')
            self.ax.set_ylabel('True Positive Rate')
            self.ax.set_title(f'Receiver operating characteristic- SVM- Multi-class')
            self.ax.legend(loc="lower right")
            self.canvas.draw()

            classes = np.unique(y)
            for i in range(len(classes)):
                class_auc = roc_auc_score(y_test[:, i], y_pred_prob[:, i])
                print(f"AUC for Class '{classes[i]}': {class_auc:.2%}")

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 1
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0

    def train_decision_tree_classifier(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.train_decision_tree_classifier()
            y_test = results['y_test']
            y_pred_prob = results['y_pred_prob']
            accuracy = results['accuracy']
            conf_matrix = results['conf_matrix']
            fpr_0, tpr_0, _ = roc_curve(y_test, y_pred_prob[:, 0])
            roc_auc_0 = auc(fpr_0, tpr_0)
            fpr_1, tpr_1, _ = roc_curve(y_test, y_pred_prob[:, 1])
            roc_auc_1 = auc(fpr_1, tpr_1)
            self.ax.clear()
            self.ax.plot(fpr_0, tpr_0, color='darkorange', lw=2, label='AUC(Class 0) (area = %0.2f)' % roc_auc_0)
            self.ax.plot(fpr_1, tpr_1, color='green', lw=2, label='AUC(Class 1) (area = %0.2f)' % roc_auc_1)
            self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.ax.set_xlabel('False Positive Rate')
            self.ax.set_ylabel('True Positive Rate')
            self.ax.set_title(
                f'Receiver operating characteristic \nAccuracy: {accuracy:.2f} , Confusion Matrix: \n{conf_matrix}')
            self.ax.legend(loc="lower right")
            self.canvas.draw()

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 1
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0

    def train_decision_tree_classifier_multiclass(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.train_decision_tree_classifier_multiclass()
            fpr = results['fpr']
            tpr = results['tpr']
            roc_auc = results['roc_auc']

            self.ax.clear()
            for i in range(len(fpr)):
                self.ax.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.ax.set_xlabel('False Positive Rate')
            self.ax.set_ylabel('True Positive Rate')
            self.ax.set_title('Receiver Operating Characteristic Curve - Multi-Class Classification - Decision Tree')
            self.ax.legend(loc='lower right')
            self.canvas.draw()

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 1
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0

    def train_neural_network(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.train_neural_network()
            y_test = results['y_test']
            test_predictions = results['test_predictions']
            test_accuracy = results['test_accuracy']
            self.ax.clear()

            self.ax.scatter(y_test, test_predictions, color='green', label='Predicted')
            self.ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red',
                         linewidth=2)
            self.ax.set_xlabel('Actual Values')
            self.ax.set_ylabel('Predicted Values')
            self.ax.set_title(
                f'Testing Data - Actual vs Predicted Values- Neural Network \nAccuracy: {test_accuracy:.2f}')
            self.ax.legend()
            self.canvas.draw()

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 1
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 0

    def train_neural_network_classifier(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.train_neural_network_classifier()
            y_test = results['y_test']
            y_test_pred = results['y_test_pred']

            fpr, tpr, _ = roc_curve(y_test, y_test_pred)
            roc_auc = auc(fpr, tpr)

            self.ax.clear()
            self.ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.ax.set_xlim([0.0, 1.0])
            self.ax.set_ylim([0.0, 1.05])
            self.ax.set_xlabel('False Positive Rate')
            self.ax.set_ylabel('True Positive Rate')
            self.ax.set_title('Receiver Operating Characteristic - Neural Network Classifier')
            self.ax.legend(loc="lower right")
            self.canvas.draw()

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 1
            self.cmp_plot = 0
            self.cmpm_plot = 0

    def compare_models_performance(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.compare_models_performance()

            fpr_rf = results['fpr_rf']
            tpr_rf = results['tpr_rf']
            auc_rf = results['auc_rf']

            fpr_svm = results['fpr_svm']
            tpr_svm = results['tpr_svm']
            auc_svm = results['auc_svm']

            fpr_dt = results['fpr_dt']
            tpr_dt = results['tpr_dt']
            auc_dt = results['auc_dt']

            fpr_nn = results['fpr_nn']
            tpr_nn = results['tpr_nn']
            auc_nn = results['auc_nn']

            self.ax.clear()
            self.ax.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Random Forest (AUC = {auc_rf:.2f})')
            self.ax.plot(fpr_svm, tpr_svm, color='green', lw=2, label=f'SVM (AUC = {auc_svm:.2f})')
            self.ax.plot(fpr_dt, tpr_dt, color='blue', lw=2, label=f'Decision Tree (AUC = {auc_dt:.2f})')
            self.ax.plot(fpr_nn, tpr_nn, color='red', lw=2, label=f'Neural Network (AUC = {auc_nn:.2f})')
            self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.ax.set_xlabel('False Positive Rate')
            self.ax.set_ylabel('True Positive Rate')
            self.ax.set_title('Receiver Operating Characteristic Curve - Random Forest vs SVM vs Decision Tree vs Neural Network')
            self.ax.legend(loc="lower right")
            self.canvas.draw()

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 1
            self.cmpm_plot = 0

    def compare_models_performance_multiclass(self):
        if self.feedback_hhc:
            results = self.feedback_hhc.compare_models_performance_multiclass()

            y = results['y']

            fpr_rf = results['fpr_rf']
            tpr_rf = results['tpr_rf']
            roc_auc_rf = results['roc_auc_rf']

            fpr_svm = results['fpr_svm']
            tpr_svm = results['tpr_svm']
            roc_auc_svm = results['roc_auc_svm']

            fpr_dt = results['fpr_dt']
            tpr_dt = results['tpr_dt']
            roc_auc_dt = results['roc_auc_dt']

            self.ax.clear()
            for i in range(len(np.unique(y))):
                self.ax.plot(fpr_rf[i], tpr_rf[i], lw=2, label=f'Random Forest Class {i} (AUC = {roc_auc_rf[i]:.2f})')
                self.ax.plot(fpr_svm[i], tpr_svm[i], lw=2, label=f'SVM Class {i} (AUC = {roc_auc_svm[i]:.2f})')
                self.ax.plot(fpr_dt[i], tpr_dt[i], lw=2, label=f'Decision Tree Class {i} (AUC = {roc_auc_dt[i]:.2f})')

            self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            self.ax.set_xlabel('False Positive Rate')
            self.ax.set_ylabel('True Positive Rate')
            self.ax.set_title(
                'Receiver Operating Characteristic Curve - Random Forest vs SVM vs Decision Tree (Multi-Class)')
            self.ax.legend(loc="lower right")
            self.canvas.draw()

            self.pca_plot = 0
            self.rf_plot = 0
            self.rfc_plot = 0
            self.dtc_plot = 0
            self.dtr_plot = 0
            self.dtcm_plot = 0
            self.svmc_plot = 0
            self.svmm_plot = 0
            self.nn_plot = 0
            self.nnc_plot = 0
            self.cmp_plot = 0
            self.cmpm_plot = 1


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x1000")
    app = FeedbackHHCInterfaceGUI(root)
    root.mainloop()
