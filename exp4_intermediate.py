import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'winequality-red.csv')
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'intermediate')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
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
    return X_train, X_test, y_train, y_test, feature_names


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


def plot_feature_importance(scores: np.ndarray, feature_names: list, filename: str, top_k: int = 11):
    order = np.argsort(scores)[::-1]
    top_idx = order[:top_k]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=scores[top_idx], y=np.array(feature_names)[top_idx], orient='h', palette='viridis')
    plt.xlabel('ANOVA F 值')
    plt.ylabel('特征')
    plt.title('特征重要性（F检验）')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, filename), dpi=150)
    plt.close()


def main():
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 分类报告
    report = classification_report(y_test, y_pred, digits=4)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    # 混淆矩阵（文件名与模板一致）
    plot_confusion_matrix(y_test, y_pred, 'confusion_matrix_custom.png')

    # 特征重要性（基于F检验）
    scores, pvals = f_classif(X_train, y_train)
    np.nan_to_num(scores, copy=False)
    plot_feature_importance(scores, feature_names, 'feature_importance.png')

    # 保存原始分数
    fi = {name: float(s) for name, s in zip(feature_names, scores)}
    with open(os.path.join(OUTPUT_DIR, 'feature_importance.json'), 'w', encoding='utf-8') as f:
        json.dump(fi, f, ensure_ascii=False, indent=2)

    print(report)


if __name__ == '__main__':
    main()
