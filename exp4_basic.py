import json
import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

RANDOM_STATE = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'winequality-red.csv')
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'basic')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list


def load_and_prepare() -> Dataset:
    df = pd.read_csv(DATA_PATH)
    if 'quality' not in df.columns:
        raise ValueError('CSV中未找到quality列')

    def map_quality(q: int) -> int:
        if q <= 4:
            return 0
        elif q <= 6:
            return 1
        else:
            return 2

    y = df['quality'].apply(map_quality).values
    X = df.drop(columns=['quality'])
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return Dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=feature_names)


class ManualGaussianNB:
    def __init__(self):
        self.classes_ = None
        self.theta_ = None  # 均值 (n_classes, n_features)
        self.var_ = None    # 方差 (n_classes, n_features)
        self.class_count_ = None
        self.class_log_prior_ = None
        self.eps_ = 1e-9

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[idx] = X_c.shape[0]
            self.theta_[idx, :] = X_c.mean(axis=0)
            # 方差加上极小值，避免除零
            self.var_[idx, :] = X_c.var(axis=0) + self.eps_

        # 拉普拉斯平滑的先验（简单处理）：
        smoothed = self.class_count_ + 1.0
        self.class_log_prior_ = np.log(smoothed / smoothed.sum())
        return self

    def _log_gaussian_prob(self, X):
        # 返回 shape: (n_samples, n_classes)
        n_samples, n_features = X.shape
        n_classes = self.theta_.shape[0]
        log_prob = np.zeros((n_samples, n_classes))
        constant = -0.5 * np.log(2.0 * np.pi)
        for c in range(n_classes):
            mean = self.theta_[c]
            var = self.var_[c]
            # 逐特征的高斯对数似然
            lp = constant - 0.5 * np.log(var) - ((X - mean) ** 2) / (2 * var)
            log_prob[:, c] = lp.sum(axis=1)
        return log_prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        joint_log_likelihood = self._log_gaussian_prob(X) + self.class_log_prior_
        indices = np.argmax(joint_log_likelihood, axis=1)
        return self.classes_[indices]


def plot_confusion_matrix(y_true, y_pred, filename: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['低(0)','中(1)','高(2)'], yticklabels=['低(0)','中(1)','高(2)'])
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('朴素贝叶斯分类器混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, filename), dpi=150)
    plt.close()


def main():
    data = load_and_prepare()

    manual_nb = ManualGaussianNB().fit(data.X_train, data.y_train)
    y_pred_manual = manual_nb.predict(data.X_test)
    acc_manual = accuracy_score(data.y_test, y_pred_manual)

    sk_nb = GaussianNB()
    sk_nb.fit(data.X_train, data.y_train)
    y_pred_sk = sk_nb.predict(data.X_test)
    acc_sk = accuracy_score(data.y_test, y_pred_sk)

    # 保存混淆矩阵（采用sklearn模型的输出）
    plot_confusion_matrix(data.y_test, y_pred_sk, 'confusion_matrix_custom.png')

    summary = {
        'accuracy_manual': float(acc_manual),
        'accuracy_sklearn': float(acc_sk),
        'report_sklearn': classification_report(data.y_test, y_pred_sk, output_dict=True)
    }

    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('手写GNB 准确率:', acc_manual)
    print('sklearn GNB 准确率:', acc_sk)


if __name__ == '__main__':
    main()
