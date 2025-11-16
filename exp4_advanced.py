import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'winequality-red.csv')
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'advanced')
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
    X = df.drop(columns=['quality']).values
    feature_names = [c for c in df.columns if c != 'quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, feature_names


def plot_multiclass_roc(X_test, y_test, clf, n_classes: int, filename: str):
    # One-vs-Rest 概率
    y_score = clf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 微观与宏观
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # 宏观平均
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    plt.figure(figsize=(7, 6))
    colors = ['C0','C1','C2']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'类别 {i} (AUC={roc_auc[i]:.3f})')
    plt.plot(fpr['micro'], tpr['micro'], linestyle='--', color='gray', lw=2, label=f'微观平均 (AUC={roc_auc["micro"]:.3f})')
    plt.plot(fpr['macro'], tpr['macro'], linestyle='--', color='black', lw=2, label=f'宏观平均 (AUC={roc_auc["macro"]:.3f})')
    plt.plot([0,1],[0,1], linestyle=':', color='k')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('多类别ROC曲线（朴素贝叶斯）')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, filename), dpi=150)
    plt.close()

    # 保存数值
    with open(os.path.join(OUTPUT_DIR, 'roc_auc_nb.json'), 'w', encoding='utf-8') as f:
        json.dump({k: float(v) for k, v in roc_auc.items()}, f, ensure_ascii=False, indent=2)


def plot_calibration_curves(X_test, y_test, base_clf, filename: str):
    # 基础模型 与 校准后模型
    calib = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)
    calib.fit(X_train, y_train)

    y_prob_base = base_clf.predict_proba(X_test)
    y_prob_cal = calib.predict_proba(X_test)

    # 为简洁，仅绘制每个类别的一条可靠性曲线
    n_classes = y_prob_base.shape[1]
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        y_true_bin = (y_test == i).astype(int)
        prob_true_b, prob_pred_b = calibration_curve(y_true_bin, y_prob_base[:, i], n_bins=10)
        prob_true_c, prob_pred_c = calibration_curve(y_true_bin, y_prob_cal[:, i], n_bins=10)
        plt.plot(prob_pred_b, prob_true_b, marker='o', label=f'未校准 类别{i}')
        plt.plot(prob_pred_c, prob_true_c, marker='s', label=f'校准后 类别{i}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='理想')
    plt.xlabel('预测概率')
    plt.ylabel('真实概率')
    plt.title('概率校准曲线（朴素贝叶斯 + Platt Scaling）')
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, filename), dpi=150)
    plt.close()


def plot_model_comparison_roc(X_train, X_test, y_train, y_test, filename: str):
    models = {
        '朴素贝叶斯': GaussianNB(),
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        '随机森林': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        'SVC': SVC(probability=True, random_state=RANDOM_STATE)
    }

    plt.figure(figsize=(7, 6))
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=[0,1,2])
        # 宏平均AUC
        fpr = []
        tpr = []
        for i in range(3):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], proba[:, i])
            fpr.append(fpr_i); tpr.append(tpr_i)
        all_fpr = np.unique(np.concatenate(fpr))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, lw=2, label=f'{name} (宏AUC={macro_auc:.3f})')

    plt.plot([0,1],[0,1], linestyle=':', color='k')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('多模型宏平均ROC曲线比较')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, filename), dpi=150)
    plt.close()


def main():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    # 朴素贝叶斯多类ROC
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    plot_multiclass_roc(X_test, y_test, nb, n_classes=3, filename='roc_curves_custom.png')

    # 概率校准曲线
    plot_calibration_curves(X_test, y_test, nb, filename='calibration_curves.png')

    # 多模型比较
    plot_model_comparison_roc(X_train, X_test, y_train, y_test, filename='model_comparison_roc.png')


if __name__ == '__main__':
    main()
