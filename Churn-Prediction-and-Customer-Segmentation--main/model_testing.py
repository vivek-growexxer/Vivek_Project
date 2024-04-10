import os
import numpy as np
import pandas as pd

from time import time

from scipy.stats import ks_2samp

# from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, KMeansSMOTE, SMOTEN, SMOTENC, SVMSMOTE
# from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier


from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix

# Data preparation and Evaluation

def features_target_split(df, target_col='Exited'):
    """
    Split the DataFrame into features and target variables.
    
    Parameters:
        df (DataFrame): The input DataFrame.
        target_col (str): The name of the target column. Default is 'Exited'.
        
    Returns:
        x (DataFrame): The features.
        y (Series): The target variable.
    """
    # Drop the target column from the DataFrame to get the features
    x = df.drop(target_col, axis=1)
    
    # Assign the target column as the y variable
    y = df[target_col]
    
    # Return the features and target variables
    return x,y


def train_test_split(x,y,df,target_col='Exited', test_size=0.2, random_state=42):
    """
    Split the features and target variables into training and testing sets.
    
    Parameters:
        x (DataFrame): The features.
        y (Series): The target variable.
        df (DataFrame): The original DataFrame.
        target_col (str): The name of the target column. Default is 'Exited'.
        test_size (float or int): The proportion or absolute number of samples to include in the testing set. Default is 0.2.
        random_state (int): The seed used by the random number generator. Default is 42.
        
    Returns:
        x_train (DataFrame): The training set features.
        x_test (DataFrame): The testing set features.
        y_train (Series): The training set target variable.
        y_test (Series): The testing set target variable.
    """
    from sklearn.model_selection import train_test_split
    
    # Split the features and target variables into training and testing sets
    # Stratified is being used to maintain the proportion of class [0 and 1] in splits.
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state, 
                                                        stratify=df[target_col])
    
    return x_train, x_test, y_train, y_test


def prediction(model, x_train, x_test):
    """
    Generate predictions using a trained logistic regression model.
    
    Parameters:
        log_reg_model (LogisticRegression): The trained logistic regression model.
        x_train (array-like or sparse matrix): The training set features.
        x_test (array-like or sparse matrix): The testing set features.
        
    Returns:
        y_pred_train (array-like): Predicted labels for the training set.
        y_pred_test (array-like): Predicted labels for the testing set.
        y_pred_test_proba (array-like): Predicted probabilities for the testing set.
    """
    # Generate predictions for the training set
    y_pred_train = model.predict(x_train)
    
    # Generate predictions for the testing set
    y_pred_test = model.predict(x_test)
    
    # Generate predicted probabilities for the testing set
    y_pred_test_proba = model.predict_proba(x_test)
    
    return y_pred_train, y_pred_test, y_pred_test_proba


class Evaluation():
    def __init__(self,y_train, y_test, y_pred_train, y_pred_test, y_pred_test_proba):
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test
        self.y_pred_test_proba = y_pred_test_proba
    
    def __ks_stats_value__(self):
        """
        Calculate the Kolmogorov-Smirnov (KS) statistic and p-value.
        
        Returns:
            ks_stat (float): The KS statistic.
            p_value (float): The p-value.
        """
        
#         # proba_non_churn contains the predicted probabilities for instances that did not churn
        proba_non_churn = self.y_pred_test_proba[:,0][self.y_test==0]
        
#         # proba_churn contains the predicted probabilities for instances that actually churned
        proba_churn = self.y_pred_test_proba[:,0][self.y_test==1]
        
        # proba_non_churn contains the predicted probabilities for instances that did not churn
#         proba_non_churn = self.y_pred_test_proba[self.y_test==0]
        
        # proba_churn contains the predicted probabilities for instances that actually churned
#         proba_churn = self.y_pred_test_proba[self.y_test==1]
        
        # Calculating Kolmogorov-Smirnov (KS) statistic and p-value
        ks_stat, p_value = ks_2samp(proba_non_churn, proba_churn)
        return ks_stat, p_value
    
    def __accuracy_value__(self):
        train_accuracy = accuracy_score(self.y_train, self.y_pred_train)
        test_accuracy = accuracy_score(self.y_test, self.y_pred_test)
        return train_accuracy, test_accuracy

    def __prec_rec_f1_value__(self, pos_label):
        """
        Calculate precision, recall, and F1-score for a given label.
        
        Parameters:
            pos_label: The label for which metrics are calculated.
        
        Returns:
            precision (float): Precision score.
            recall (float): Recall score.
            f1 (float): F1-score.
        """
        precision = precision_score(self.y_test, self.y_pred_test,pos_label=pos_label)
        recall = recall_score(self.y_test, self.y_pred_test,pos_label=pos_label)
        f1 = f1_score(self.y_test, self.y_pred_test, pos_label=pos_label)
        return precision, recall, f1

    def __roc_value__(self):
        roc_auc = roc_auc_score(self.y_test, self.y_pred_test)
        return roc_auc

    def __confusion_matrix_value__(self):
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred_test).ravel()
        return tn, fp, fn, tp
    
    def main(self):
        train_accuracy, test_accuracy = self.__accuracy_value__()
        
        precision_0, recall_0, f1_0 = self.__prec_rec_f1_value__(pos_label=0)
        precision_1, recall_1, f1_1 = self.__prec_rec_f1_value__(pos_label=1)
        
        ks_stat, p_value = self.__ks_stats_value__()
        
        roc_auc = self.__roc_value__()
        
        tn, fp, fn, tp = self.__confusion_matrix_value__()
        
        all_metrics = [train_accuracy, test_accuracy, roc_auc, 
                       precision_0, recall_0, f1_0, 
                       precision_1, recall_1, f1_1, 
                       ks_stat, p_value, 
                       tp, tn, fp, fn]
        
        all_metrics = [round(value, ndigits=6) for value in all_metrics]
        all_metrics_dict = {'train_acc':all_metrics[0], 'test_acc':all_metrics[1], 'roc_auc':all_metrics[2],  
                            'class_0':{'precision':all_metrics[3], 'recall':all_metrics[4], 'f1':all_metrics[5]}, 
                            'class_1':{'precision':all_metrics[6], 'recall':all_metrics[7], 'f1':all_metrics[8]},
                            'ks_stats':all_metrics[9], 'p_value':all_metrics[10],
                            'tp':all_metrics[11],'tn':all_metrics[12],'fp':all_metrics[13],'fn':all_metrics[14]}
        
        return all_metrics, all_metrics_dict
    
def logistic_model_train(x_train, y_train, random_state=42, max_iter=1000):
    """
    Train a logistic regression model using the provided training data.
    
    Parameters:
        x_train (DataFrame): The training set features.
        y_train (Series): The training set target variable.
        random_state (int): The seed used by the random number generator. Default is 42.
        max_iter (int): The maximum number of iterations for the solver to converge. Default is 1000.
        
    Returns:
        log_reg_model (LogisticRegression): The trained logistic regression model.
    """
    
    # Create an instance of LogisticRegression model with specified random_state and max_iter
    log_reg_model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    
    # Fit the logistic regression model to the training data
    log_reg_model.fit(x_train, y_train)
    
    return log_reg_model


def gnb_model_train(x_train, y_train):
    
    # instantiate the model
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    return gnb

def svc_model_train(x_train, y_train, random_state=42):

    svc = SVC(probability=True,random_state=random_state)
    svc.fit(x_train, y_train)
    return svc

def adaboost_model_train(x_train, y_train, random_state=42):

    adb_model = AdaBoostClassifier(random_state=random_state)
    adb_model.fit(x_train, y_train)
    return adb_model

def etc_model_train(x_train, y_train, random_state=42):
    etc_model = ExtraTreesClassifier(random_state=random_state)
    etc_model.fit(x_train, y_train)
    return etc_model

def gbc_model_train(x_train, y_train, random_state=42):
    gbc_model = GradientBoostingClassifier(random_state=random_state)
    gbc_model.fit(x_train, y_train)
    return gbc_model

def hgbc_model_train(x_train, y_train, random_state=42):
    hgbc_model = HistGradientBoostingClassifier(random_state=random_state)
    hgbc_model.fit(x_train, y_train)
    return hgbc_model

def rfc_model_train(x_train, y_train, random_state=42):
    rfc_model = RandomForestClassifier(random_state=random_state)
    rfc_model.fit(x_train, y_train)
    return rfc_model

def bbc_model_train(x_train, y_train, random_state=42):
    bbc_model = BalancedBaggingClassifier(random_state=random_state)
    bbc_model.fit(x_train, y_train)
    return bbc_model

def brfc_model_train(x_train, y_train, random_state=42):
    brfc_model = BalancedRandomForestClassifier(random_state=random_state)
    brfc_model.fit(x_train, y_train)
    return brfc_model

def eec_model_train(x_train, y_train, random_state=42):
    eec_model = EasyEnsembleClassifier(random_state=random_state)
    eec_model.fit(x_train, y_train)
    return eec_model

def lgbm_model_train(x_train, y_train, random_state=42):
    lgbm_model = LGBMClassifier(random_state=random_state)
    lgbm_model.fit(x_train, y_train)
    return lgbm_model

def catboost_model_train(x_train, y_train, random_state=42):
    catboost_model = CatBoostClassifier(random_state=random_state)
    catboost_model.fit(x_train, y_train, verbose=False)
    return catboost_model

def train_all_models(x_train, y_train, model_name):
    
    if model_name == 'logistic_regression':
        model = logistic_model_train(x_train, y_train)
        
    elif model_name == 'gaussian_naive_bayes':
        model = gnb_model_train(x_train, y_train)
        
    elif model_name == 'support_vector_classifier':
        model = svc_model_train(x_train, y_train)
        
    elif model_name == 'ada_boost':
        model = adaboost_model_train(x_train, y_train)
        
    elif model_name == 'extra_trees_classifier':
        model = etc_model_train(x_train, y_train)

    elif model_name == 'gradient_boosting_classifier':
        model = gbc_model_train(x_train, y_train)
    
    elif model_name == 'hist_gradient_boosting_classifier':
        model = hgbc_model_train(x_train, y_train)
    
    elif model_name == 'random_forest_classifier':
        model = rfc_model_train(x_train, y_train)

    elif model_name == 'balanced_bagging_classifier':
        model = bbc_model_train(x_train, y_train)
        
    elif model_name == 'balanced_random_forest_classifier':
        model = brfc_model_train(x_train, y_train)
        
    elif model_name == 'easy_ensemble_classifier':
        model = eec_model_train(x_train, y_train)

    elif model_name == 'lgbm_classifier':
        model = lgbm_model_train(x_train, y_train)

    elif model_name == 'catboost_classifier':
        model = catboost_model_train(x_train, y_train)

    else:
        print("Check model name")
    
    return model



