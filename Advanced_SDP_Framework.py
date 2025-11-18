import pandas as pd
import warnings
import os
import joblib  # 用于保存和加载模型
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, 
    matthews_corrcoef, 
    roc_auc_score, 
    confusion_matrix,
    accuracy_score
)

# --- 导入所有模型 ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# --- 配置 ---
MODEL_DIR = 'saved_models_v3'  # 模型保存的文件夹名称 (避免跟V2的混淆)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- 步骤 1: 数据收集 (Data Collection) ---
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, na_values=["?"]) 
        target_col_name = None
        if 'defects' in df.columns: target_col_name = 'defects'
        elif 'bug' in df.columns: target_col_name = 'bug'
        elif 'problems' in df.columns: target_col_name = 'problems'
            
        if target_col_name:
            target_series = df[target_col_name]
            df = df.drop(columns=[target_col_name])
        else:
            print(f"错误：在 {filepath} 中未找到 'defects', 'bug' 或 'problems' 列。")
            return None

        df = df.apply(pd.to_numeric, errors='coerce')
        df['defects'] = target_series

        if df['defects'].dtype == 'object':
            df['defects'] = df['defects'].map({'yes': True, 'no': False})
        
        df['defects'] = (df['defects'] > 0).astype(bool)
        return df
    except Exception as e:
        print(f"加载 {filepath} 时出错: {e}")
        return None

# --- 步骤 2: 数据预处理 (Data Preprocessing) ---
def preprocess_data(df):
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"   -> {df.name}: 发现缺失值，使用中位数填充 {len(missing_cols)} 列。")
        for col in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())

    X = df.drop('defects', axis=1)
    y = df['defects']
    if len(y.value_counts()) < 2: raise ValueError("单一类别")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# --- 模型注册表 ---
def get_models_and_grids():
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'SVC': SVC(random_state=42, probability=True), 
        'GaussianNB': GaussianNB(),
        'KNeighbors': KNeighborsClassifier(n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', use_label_encoder=False),
        'MLPClassifier': MLPClassifier(random_state=42, max_iter=500)
    }
    param_grids = {
        'LogisticRegression': {'class_weight': ['balanced'], 'C': [0.1, 1.0, 10]},
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10], 'class_weight': ['balanced']},
        'SVC': {'C': [1.0, 10], 'kernel': ['rbf'], 'class_weight': ['balanced']},
        'GaussianNB': {'var_smoothing': [1e-9, 1e-8]},
        'KNeighbors': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'scale_pos_weight': [1, 5, 10]},
        'MLPClassifier': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001]}
    }
    return models, param_grids

# --- 步骤 3: 模型构建 (带GridSearch) ---
def build_and_train_model(X_train, y_train, base_model, param_grid):
    grid = GridSearchCV(base_model, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    try:
        grid.fit(X_train, y_train)
        return grid.best_estimator_
    except Exception:
        return None

# --- 步骤 5: 评估 ---
def evaluate_model(y_test, y_pred, y_pred_proba, dataset_name, model_name):
    cm = confusion_matrix(y_test, y_pred)
    # 处理混淆矩阵形状以防万一
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0 # 简化处理极端情况

    return {
        'Dataset': dataset_name,
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

# --- 主函数 ---
def main():
    print("=== 开始执行软件缺陷预测框架  ===")
    
    dataset_files = ['cm1.csv', 'jm1.csv', 'kc1.csv', 'kc2.csv', 'pc1.csv']
    models_to_run, param_grids_to_run = get_models_and_grids()
    all_results = []

    for dataset_file in dataset_files:
        print(f"--- 正在处理数据集: {dataset_file} ...")
        
        df = load_data(dataset_file)
        if df is None: continue
        df.name = dataset_file

        try:
            X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        except ValueError: continue

        for model_name in models_to_run:
            # 1. 定义模型路径
            model_filename = os.path.join(MODEL_DIR, f"{dataset_file}_{model_name}.pkl")
            model = None
            
            # 2. 智能加载
            if os.path.exists(model_filename):
                # print(f"   (已加载 {model_name})") # 可选：取消注释以查看加载提示
                model = joblib.load(model_filename)
            else:
                # 3. 训练并保存
                # print(f"   (正在训练 {model_name}...)") # 可选
                base_model = models_to_run[model_name]
                param_grid = param_grids_to_run[model_name]
                model = build_and_train_model(X_train, y_train, base_model, param_grid)
                
                if model is not None:
                    joblib.dump(model, model_filename)
            
            if model is None: continue
            
            # 4. 预测与评估
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results = evaluate_model(y_test, y_pred, y_pred_proba, dataset_file, model_name)
            all_results.append(results)

    print("\n\n" + "="*30 + " 最终评估摘要 " + "="*30)
    
    if not all_results:
        print("未成功运行任何实验。")
        return
        
    results_df = pd.DataFrame(all_results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    try:
        # 打印4个表格
        metrics = ['Accuracy', 'F1-Score', 'MCC', 'ROC-AUC']
        for metric in metrics:
            print(f"\n摘要: {metric} ")
            print(results_df.pivot_table(index='Dataset', columns='Model', values=metric).to_markdown(floatfmt=".4f"))

    except ImportError:
        print("\n*** 警告: 未安装 'tabulate'。无法打印 Markdown 表格。 ***")
    except Exception as e:
        print(f"打印摘要时出错: {e}")

    results_df.to_csv("full_experiment_results_v3.csv", index=False)
    print(f"\n完整结果已保存到 'full_experiment_results_v3.csv'")
    print("\n=== 软件缺陷预测框架执行完毕 ===")

if __name__ == "__main__":
    main()