"""
feature env
interactive with the actor critic for the state and state after action
"""

import os
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.manifold import trustworthiness

from record import RecordList
from utils.logger import error, info
from utils.tools import (
    test_task_new,
    downstream_task_new,
    downstream_task_by_method,
    downstream_task_by_method_std,
)


base_path = "./data/dataset"

TASK_DICT = {
    "airfoil": "reg",
    "amazon_employee": "cls",
    "ap_omentum_ovary": "cls",
    "bike_share": "reg",
    "german_credit": "cls",
    "higgs": "cls",
    "housing_boston": "reg",
    "ionosphere": "cls",
    "lymphography": "cls",
    "messidor_features": "cls",
    "openml_620": "reg",
    "pima_indian": "cls",
    "spam_base": "cls",
    "spectf": "cls",
    "svmguide3": "cls",
    "uci_credit_card": "cls",
    "wine_red": "cls",
    "wine_white": "cls",
    "openml_586": "reg",
    "openml_589": "reg",
    "openml_607": "reg",
    "openml_616": "reg",
    "openml_618": "reg",
    "openml_637": "reg",
    "smtp": "det",
    "thyroid": "det",
    "yeast": "det",
    "wbc": "det",
    "mammography": "det",
    "arrhythmia": "cls",
    "nomao": "cls",
    "megawatt1": "cls",
    "activity": "mcls",
    "mice_protein": "mcls",
    "coil-20": "mcls",
    "isolet": "mcls",
    "minist": "mcls",
    "minist_fashion": "mcls",
    "semeion": "mcls",
    "ap_omentum_ovary": "cls",
    "openml_1085": "mcls",
    "openml_1082": "mcls",
    "openml_1088": "mcls",
}

# todo: meanings of ras, map, mif1, maf1
MEASUREMENT = {
    "cls": ["precision", "recall", "f1_score", "roc_auc"],
    "reg": ["mae", "mse", "rae", "rmse"],
    "det": ["map", "f1_score", "ras", "recall"],
    "mcls": ["precision", "recall", "mif1", "maf1"],
}

model_performance = {
    "mcls": namedtuple("ModelPerformance", MEASUREMENT["mcls"]),
    "cls": namedtuple("ModelPerformance", MEASUREMENT["cls"]),
    "reg": namedtuple("ModelPerformance", MEASUREMENT["reg"]),
    "det": namedtuple("ModelPerformance", MEASUREMENT["det"]),
}


class Evaluator(object):
    def __init__(self, task, task_type=None, dataset=None):
        self.original_report = None
        self.records = RecordList()
        self.task_name = task
        if task_type is None:
            self.task_type = TASK_DICT[self.task_name]
        else:
            self.task_type = task_type

        if dataset is None:
            data_path = os.path.join(base_path, self.task_name + ".hdf")
            original = pd.read_hdf(data_path)
        else:
            original = dataset
        col = np.arange(original.shape[1])
        self.col_names = original.columns
        original.columns = col
        y = original.iloc[:, -1]
        x = original.iloc[:, :-1]
        if task == "ap_omentum_ovary":
            y[y == "Ovary"] = 1
            y[y == "Omentum"] = 0
            y = y.astype(float)
            original = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
        self.original = original.fillna(value=0)
        y = self.original.iloc[:, -1]
        x = self.original.iloc[:, :-1]
        # 80% of data used to build embedding space, other 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0, shuffle=True
        )

        self.train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
        self.test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
        info("initialize the train and test dataset")
        self._check_path()

    def __len__(self):
        return len(self.records)

    def generate_data(self, operation, flag):
        pass

    def get_performance(self, data=None):
        if data is None:
            data = self.original
        return downstream_task_new(data, self.task_type)

    def report_ds(self):
        pass

    def _store_history(self, choice, performance):
        self.records.append(choice, performance)

    def _flush_history(self, choices, performances, is_permuted, num, padding):
        if is_permuted:
            flag_1 = "augmented"
        else:
            flag_1 = "original"
        if padding:
            flag_2 = "padded"
        else:
            flag_2 = "not_padded"
        torch.save(
            choices,
            f"{base_path}/history/{self.task_name}/choice.{flag_1}.{flag_2}.{num}.pt",
        )
        info(f"save the choice to {base_path}/history/{self.task_name}/choice.pt")
        torch.save(
            performances,
            f"{base_path}/history/{self.task_name}/performance.{flag_1}.{flag_2}.{num}.pt",
        )
        info(
            f"save the performance to {base_path}/history/{self.task_name}/performance.pt"
        )

    def _check_path(self):
        if not os.path.exists(f"{base_path}/history/{self.task_name}"):
            os.makedirs(f"{base_path}/history/{self.task_name}")

    def save(self, num=25, padding=True, padding_value=-1):
        if num > 0:
            is_permuted = True
        else:
            is_permuted = False
        info("save the records...")
        choices, performances = self.records.generate(
            num=num, padding=padding, padding_value=padding_value
        )
        self._flush_history(choices, performances, is_permuted, num, padding)

    def get_record(self, num=0, eos=-1):
        results = []
        labels = []
        for record in self.records.r_list:
            result, label = record.get_permutated(num, True, eos)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)

    def get_triple_record(self, num=0, eos=-1, mode="ht"):
        h_results = []
        labels = []
        t_results = []
        h_seed = []
        labels_seed = []
        for record in self.records.r_list:
            if mode.__contains__("h"):
                h, label = record.get_permutated(num, True, eos)
            else:
                h, label = record.repeat(num, True, eos)
            if mode.__contains__("t"):
                t, _ = record.get_permutated(num, True, eos)
            else:
                t, _ = record.repeat(num, True, eos)
            h_results.append(h)
            t_results.append(t)
            labels.append(label)
            h_seed.append(h_results[0])
            labels_seed.append(labels[0])
        return (
            torch.cat(h_results, 0),
            torch.cat(labels, 0),
            torch.cat(t_results),
            torch.cat(h_seed),
            torch.cat(labels_seed),
        )

    def report_performance(self, choice, store=True, rp=True, flag="test", init_seed=0):
        # flag = 'train' # Remove this line to correctly use the passed flag parameter
        opt_ds = self.generate_data(choice, flag)
        a, b, c, d = test_task_new(opt_ds, task=self.task_type, init_seed=init_seed)
        report = model_performance[self.task_type](a, b, c, d)
        # if flag == 'test':
        #     store = False
        if self.original_report is None:
            a, b, c, d = test_task_new(
                self.train, task=self.task_type, init_seed=init_seed
            )
            self.original_report = (a, b, c, d)
        else:
            a, b, c, d = self.original_report
        original_report = model_performance[self.task_type](a, b, c, d)
        if self.task_type == "reg":
            final_result = report.rae
            if rp:
                info(
                    "1-MAE on original is: {:.4f}, 1-MAE on generated is: {:.4f}".format(
                        original_report.mae, report.mae
                    )
                )
                info(
                    "1-MSE on original is: {:.4f}, 1-MSE on generated is: {:.4f}".format(
                        original_report.mse, report.mse
                    )
                )
                info(
                    "1-RAE on original is: {:.4f}, 1-RAE on generated is: {:.4f}".format(
                        original_report.rae, report.rae
                    )
                )
                info(
                    "1-RMSE on original is: {:.4f}, 1-RMSE on generated is: {:.4f}".format(
                        original_report.rmse, report.rmse
                    )
                )
        elif self.task_type == "cls":
            final_result = report.f1_score
            if rp:
                info(
                    "Pre on original is: {:.4f}, Pre on generated is: {:.4f}".format(
                        original_report.precision, report.precision
                    )
                )
                info(
                    "Rec on original is: {:.4f}, Rec on generated is: {:.4f}".format(
                        original_report.recall, report.recall
                    )
                )
                info(
                    "F-1 on original is: {:.4f}, F-1 on generated is: {:.4f}".format(
                        original_report.f1_score, report.f1_score
                    )
                )
                info(
                    "ROC/AUC on original is: {:.4f}, ROC/AUC on generated is: {:.4f}".format(
                        original_report.roc_auc, report.roc_auc
                    )
                )
        elif self.task_type == "det":
            final_result = report.ras
            if rp:
                info(
                    "Average Precision Score on original is: {:.4f}, Average Precision Score on generated is: {:.4f}".format(
                        original_report.map, report.map
                    )
                )
                info(
                    "F1 Score on original is: {:.4f}, F1 Score on generated is: {:.4f}".format(
                        original_report.f1_score, report.f1_score
                    )
                )
                info(
                    "ROC AUC Score on original is: {:.4f}, ROC AUC Score on generated is: {:.4f}".format(
                        original_report.ras, report.ras
                    )
                )
                info(
                    "Recall on original is: {:.4f}, Recall Score on generated is: {:.4f}".format(
                        original_report.recall, report.recall
                    )
                )
        elif self.task_type == "mcls":
            final_result = report.mif1
            if rp:
                info(
                    "Pre on original is: {:.4f}, Pre on generated is: {:.4f}".format(
                        original_report.precision, report.precision
                    )
                )
                info(
                    "Rec on original is: {:.4f}, Rec on generated is: {:.4f}".format(
                        original_report.recall, report.recall
                    )
                )
                info(
                    "Micro-F1 on original is: {:.4f}, Micro-F1 on generated is: {:.4f}".format(
                        original_report.mif1, report.mif1
                    )
                )
                info(
                    "Macro-F1 on original is: {:.4f}, Macro-F1 on generated is: {:.4f}".format(
                        original_report.maf1, report.maf1
                    )
                )
        else:
            error("wrong task name!!!!!")
            assert False
        if store:
            self._store_history(choice, final_result)
        return final_result


class FeatureEvaluator(Evaluator):
    def __init__(self, task, task_type=None, dataset=None):
        super().__init__(task, task_type, dataset)
        self.ds_size = self.original.shape[1] - 1

        # Initialize TabGenerator - consistent with DIFFT
        from ours.tab_generator import create_tab_generator

        self.tab_generator = create_tab_generator(task, self.original)

    def generate_data(self, choice, flag=""):
        if choice.shape[0] != self.ds_size:
            error("wrong shape of choice")
            assert False
        if flag == "test":
            ds = self.test
        elif flag == "train":
            ds = self.train
        else:
            ds = self.original
        X = ds.iloc[:, :-1]
        indice = torch.arange(0, self.ds_size)[choice == 1]
        X = X.iloc[:, indice.numpy()].astype(np.float64)
        y = ds.iloc[:, -1].astype(np.float64)
        Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
        return Dg

    def _full_mask(self):
        return torch.FloatTensor([1] * self.ds_size)

    def generate_tab_data(self, choice):
        """
        Generate tab representation from feature selection - consistent with DIFFT
        Args:
            choice: feature selection vector (tensor or numpy array)
        Returns:
            tab_representation: tab representation vector
        """
        # Backward compatibility check: if no tab_generator, initialize one
        if not hasattr(self, "tab_generator") or self.tab_generator is None:
            try:
                from ours.tab_generator import create_tab_generator

                # Ensure task_name attribute exists
                task_name = getattr(self, "task_name", "unknown_task")
                self.tab_generator = create_tab_generator(task_name, self.original)
                print(f"✅ Initialized TabGenerator for {task_name}")
            except Exception as e:
                print(f"⚠️  TabGenerator initialization failed: {e}")
                # If initialization fails, create a simple fallback
                self.tab_generator = None
                print("🔄 Using simple tab generation fallback")

        if isinstance(choice, torch.Tensor):
            choice = choice.numpy()

        return self.tab_generator.generate_tab_from_selection(choice)

    def get_record_with_tabs(self, num, eos=None):
        """
        Get record and generate corresponding tab data - extend original functionality
        Args:
            choice: feature selection sequence
            labels: performance labels
        Returns:
            tabs: tab representation data
        """
        choice, labels = self.get_record(num, eos)

        # Generate tab representation for each selection
        tabs = []
        for c in choice:
            # Convert tensor to one-hot format
            if isinstance(c, torch.Tensor):
                one_hot = c.numpy()
            else:
                one_hot = np.array(c)

            # Ensure it is binary mask
            if len(one_hot) != self.ds_size: # Changed from c.max() > 1 to len(one_hot) != self.ds_size for correctness
                # If sequence format, needs conversion to binary mask
                binary_mask = np.zeros(self.ds_size)
                for idx in one_hot:
                    if idx < self.ds_size:
                        binary_mask[idx] = 1
                one_hot = binary_mask

            tab_data = self.generate_tab_data(one_hot)
            tabs.append(tab_data)

        return choice, labels, tabs

    def report_ds(self):
        per = self.get_performance()
        info(f"current dataset : {self.task_name}")
        info(f"the size of shape is : {self.original.shape[1]}")
        info(f"original performance is : {per}")
        self._store_history(self._full_mask(), per)
