import pandas as pd
import numpy as np
import warnings
import os
import joblib  # 用于保存模型
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, 
    matthews_corrcoef, 
    roc_auc_score, 
    accuracy_score
)

# --- 9 个高级模型 ---
from sklearn.ensemble import (
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    ExtraTreesClassifier, 
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier 

warnings.filterwarnings('ignore')

# --- 配置保存路径 ---
MODEL_DIR = 'saved_models_baseline'  # 基准模型保存位置
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- 步骤 1: 数据收集 ---
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, na_values=["?"]) 
        target_col_name = None
        for col in ['defects', 'bug', 'problems']:
            if col in df.columns:
                target_col_name = col
                break
        
        if target_col_name:
            target_series = df[target_col_name]
            df = df.drop(columns=[target_col_name])
        else:
            return None

        df = df.apply(pd.to_numeric, errors='coerce')
        df['defects'] = target_series

        if df['defects'].dtype == 'object':
            df['defects'] = df['defects'].map({'yes': True, 'no': False})
        
        df['defects'] = (df['defects'] > 0).astype(bool)
        return df
    except Exception:
        return None

# --- 步骤 2: 数据预处理 (基准：无过采样) ---
def preprocess_data_baseline(df):
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X = df.drop('defects', axis=1)
    y = df['defects']
    
    if len(y.value_counts()) < 2: raise ValueError("单一类别")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# --- 步骤 3: 获取模型 ---
def get_new_models_and_grids():
    models = {
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_jobs=-1, random_state=42),
        'Bagging': BaggingClassifier(random_state=42, n_jobs=-1),
        'SGD': SGDClassifier(loss='log_loss', random_state=42, n_jobs=-1),
        'PassiveAggressive': PassiveAggressiveClassifier(random_state=42, n_jobs=-1),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'DeepMLP_3Layers': MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=42)
    }
    
    param_grids = {
        'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]},
        'ExtraTrees': {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]},
        'Bagging': {'n_estimators': [10, 50]}, 
        'SGD': {'alpha': [0.0001, 0.001], 'penalty': ['l2', 'l1']},
        'PassiveAggressive': {'C': [0.1, 1.0], 'loss': ['hinge', 'squared_hinge']},
        'LDA': {'solver': ['svd', 'lsqr']},
        'QDA': {'reg_param': [0.0, 0.1]},
        'DeepMLP_3Layers': {'alpha': [0.0001, 0.001], 'activation': ['relu', 'tanh']}
    }
    return models, param_grids

# --- 主流程 ---
def main():
    print("=== SDP 基准框架 (无过采样对照组 + 智能保存) ===")
    
    dataset_files = ['cm1.csv', 'jm1.csv', 'kc1.csv', 'kc2.csv', 'pc1.csv']
    models_to_run, param_grids_to_run = get_new_models_and_grids()
    all_results = []

    for dataset_file in dataset_files:
        print(f"\n--- 正在处理数据集: {dataset_file} ...")
        
        df = load_data(dataset_file)
        if df is None: continue

        try:
            X_train, X_test, y_train, y_test = preprocess_data_baseline(df)
        except Exception as e:
            print(f"   -> 预处理失败: {e}")
            continue

        for model_name in models_to_run:
            # 1. 定义模型路径
            model_filename = os.path.join(MODEL_DIR, f"{dataset_file}_{model_name}.pkl")
            best_model = None

            # 2. 智能加载
            if os.path.exists(model_filename):
                # print(f"   (已加载 {model_name})") # 可选
                best_model = joblib.load(model_filename)
            else:
                # 3. 训练并保存
                base_model = models_to_run[model_name]
                params = param_grids_to_run[model_name]
                grid = GridSearchCV(base_model, params, scoring='roc_auc', cv=3, n_jobs=-1)
                
                try:
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    joblib.dump(best_model, model_filename)
                except Exception as e:
                    print(f"   -> 模型 {model_name} 训练失败: {e}")
                    continue

            if best_model is None: continue
                
            # 4. 预测
            y_pred = best_model.predict(X_test)
            
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            elif hasattr(best_model, "decision_function"):
                 y_proba = best_model.decision_function(X_test)
                 auc = roc_auc_score(y_test, y_proba)
            else:
                 auc = 0.5 
            
            all_results.append({
                'Dataset': dataset_file,
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'MCC': matthews_corrcoef(y_test, y_pred),
                'ROC-AUC': auc
            })

    # --- 最终输出 ---
    print("\n\n" + "="*30 + " 最终评估摘要 (基准/无过采样) " + "="*30)
    if not all_results: return

    results_df = pd.DataFrame(all_results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    try:
        metrics = ['Accuracy', 'MCC', 'ROC-AUC', 'F1-Score']
        for metric in metrics:
            print(f"\n--- 摘要: {metric} (越高越好) ---")
            print(results_df.pivot_table(index='Dataset', columns='Model', values=metric).to_markdown(floatfmt=".4f"))
    except ImportError:
        print("\n*** 警告: 未安装 'tabulate'。无法打印 Markdown 表格。 ***")
        print(results_df.pivot_table(index='Dataset', columns='Model', values='MCC'))

    results_df.to_csv("baseline_results.csv", index=False)
    print(f"\n完整结果已保存到 'baseline_results.csv'")

if __name__ == "__main__":
    main()