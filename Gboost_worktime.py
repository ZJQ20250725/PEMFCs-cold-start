"""
Created on 2025.06.16

@Source code author: Zhu, J.Q

This code has been modified to generate an optimized XGBoost model with refined
Bayesian hyperparameter tuning for simulating the time required for a PEMFCs to warm up from -20℃ to 30℃ during cold-start.


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import joblib
import xgboost as xgb

# 设置全局字体和字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['legend.fontsize'] = 24

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(2345)


# ================== 自动化预处理函数 ==================
def auto_normalize(data):
    """自动识别特征/输出的归一化参数"""
    return {
        'features': {
            'min': np.min(data[:, :-1], axis=0),
            'max': np.max(data[:, :-1], axis=0)
        },
        'output': {
            'min': np.min(data[:, -1]),
            'max': np.max(data[:, -1])
        }
    }


def apply_normalization(data, norms, is_output=False):
    """应用归一化"""
    if is_output:
        return (data - norms['output']['min']) / (norms['output']['max'] - norms['output']['min'])
    else:
        return (data - norms['features']['min']) / (norms['features']['max'] - norms['features']['min'])


def denormalize(data, norms, is_output=False):
    """反归一化"""
    if is_output:
        return data * (norms['output']['max'] - norms['output']['min']) + norms['output']['min']
    else:
        return data * (norms['features']['max'] - norms['features']['min']) + norms['features']['min']


# ================== 残差分析函数 ==================
def analyze_residuals(y_true, predictions, title):
    """
    执行残差分析并生成可视化图表
    参数：
        y_true: 反归一化后的真实值
        predictions: 反归一化后的预测值
        title: 图表标题前缀
    """
    residuals = y_true - predictions

    plt.figure(figsize=(18, 6))

    # 子图1：残差-预测值散点图
    plt.subplot(1, 3, 1)
    plt.scatter(predictions, residuals, alpha=0.6, edgecolors='w', s=40)
    plt.axhline(0, color='r', linestyle='--', lw=1)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title(f'{title} - Residuals vs Predicted', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 子图2：残差分布直方图
    plt.subplot(1, 3, 2)
    sns.histplot(residuals, kde=True, color='#2ca02c')
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'{title} - Residual Distribution', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 子图3：Q-Q图
    plt.subplot(1, 3, 3)
    sm.qqplot(residuals, line='45', fit=True)
    plt.title(f'{title} - Q-Q Plot', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{title}_residuals.png', dpi=300)
    plt.show()


# ================== 贝叶斯优化XGBoost超参数 ==================
def objective(params):
    """
    贝叶斯优化的目标函数：最小化交叉验证的RMSE
    """
    learning_rate, n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree = params

    # 初始化XGBoost模型
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective='reg:squarederror',
        nthread=4,
        scale_pos_weight=1,
        seed=2345
    )

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=2345)
    rmse_scores = []

    for train_idx, val_idx in kf.split(xa):
        x_train, x_val = xa[train_idx], xa[val_idx]
        y_train, y_val = ya[train_idx].flatten(), ya[val_idx].flatten()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)

        # 反归一化计算真实误差
        y_val_true = denormalize(y_val, norms, is_output=True)
        y_pred_true = denormalize(y_pred, norms, is_output=True)

        rmse = np.sqrt(np.mean((y_val_true - y_pred_true) ** 2))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


# ================== 主程序 ==================
if __name__ == "__main__":
    # 数据加载与处理
    print("Loading data...")
    data = pd.read_csv("TRAIN_worktime.csv")
    data = np.array(data)

    # 自动归一化
    norms = auto_normalize(data)
    xa = apply_normalization(data[:, :-1], norms)
    ya = apply_normalization(data[:, [-1]], norms, is_output=True).flatten()

    # 数据分割
    permutation = np.random.permutation(xa.shape[0])
    xa = xa[permutation, :]
    ya = ya[permutation]

    n = xa.shape[0]
    x, x_val = xa[:int(n * 0.8)], xa[int(n * 0.8):]
    y, y_val = ya[:int(n * 0.8)], ya[int(n * 0.8):]

    # ================== 贝叶斯优化XGBoost超参数 ==================
    print("\n开始贝叶斯优化XGBoost超参数（优化搜索边界）...")

    # 定义XGBoost的搜索空间
    search_space = [
        Real(0.01, 0.3, name='learning_rate'),  # 学习率
        Integer(50, 300, name='n_estimators'),  # 树的数量
        Integer(3, 10, name='max_depth'),  # 最大树深度
        Real(1, 10, name='min_child_weight'),  # 最小子节点权重
        Real(0, 0.5, name='gamma'),  # 叶节点分裂阈值
        Real(0.5, 1.0, name='subsample'),  # 样本采样率
        Real(0.5, 1.0, name='colsample_bytree')  # 特征采样率
    ]

    # 执行优化
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=70,  # 迭代次数
        n_random_starts=15,  # 随机起点
        random_state=2345,
        verbose=True
    )

    # 输出最优参数
    print("\n最优XGBoost参数:")
    print(f"学习率: {result.x[0]:.4f}")
    print(f"树的数量: {int(result.x[1])}")
    print(f"最大深度: {int(result.x[2])}")
    print(f"最小子节点权重: {result.x[3]:.4f}")
    print(f"叶节点分裂阈值: {result.x[4]:.4f}")
    print(f"样本采样率: {result.x[5]:.4f}")
    print(f"特征采样率: {result.x[6]:.4f}")
    print(f"交叉验证RMSE: {result.fun:.6f}")

    # ================== 使用最优参数训练最终模型 ==================
    print("\n使用最优参数训练最终XGBoost模型...")

    # 构建最优模型
    best_model = xgb.XGBRegressor(
        learning_rate=result.x[0],
        n_estimators=int(result.x[1]),
        max_depth=int(result.x[2]),
        min_child_weight=result.x[3],
        gamma=result.x[4],
        subsample=result.x[5],
        colsample_bytree=result.x[6],
        objective='reg:squarederror',
        nthread=4,
        scale_pos_weight=1,
        seed=2345
    )

    best_model.fit(x, y)

    # 保存模型
    joblib.dump(best_model, 'worktime_XGB_optimized.pkl')
    print(f"优化后的XGBoost模型已保存: worktime_XGB_optimized.pkl")


    # ================== 结果评估 ==================
    def compute_metrics(y_true, predictions):
        """计算评估指标"""
        mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
        rmse = np.sqrt(np.mean((y_true - predictions) ** 2))
        r = np.corrcoef(y_true, predictions)[0, 1]
        r2 = r ** 2
        return mape, rmse, r, r2


    def evaluate_and_analyze(x_data, y_data, title, original_data=None):
        """综合评估和残差分析"""
        # 获取预测结果
        predictions = best_model.predict(x_data)

        # 反归一化
        y_true = denormalize(y_data, norms, is_output=True).flatten()
        predictions = denormalize(predictions, norms, is_output=True).flatten()

        # 计算指标
        mape, rmse, r, r2 = compute_metrics(y_true, predictions)

        # 打印结果
        print(f"\n{title} Results:")
        print(f"MAPE: {mape:.2f}%")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Pearson R: {r:.6f}")

        # 残差分析
        analyze_residuals(y_true, predictions, title)

        # 保存结果到CSV文件
        if original_data is not None:
            # 反归一化特征
            x_denorm = denormalize(x_data, norms)

            # 构建数据框
            df = pd.DataFrame(x_denorm, columns=[f'Feature_{i + 1}' for i in range(x_denorm.shape[1])])
            df['True_Output'] = y_true
            df['Predicted_Output'] = predictions

            # 保存到CSV文件
            filename = f"{title.lower().replace(' ', '_')}_results.csv"
            df.to_csv(filename, index=False)
            print(f"\nSaved {filename} with {len(df)} records.")

        return predictions


    print("\n" + "=" * 40)
    train_pred = evaluate_and_analyze(x, y, "Training Set", original_data=data[:int(n * 0.8), :-1])
    print("\n" + "=" * 40)
    val_pred = evaluate_and_analyze(x_val, y_val, "Validation Set", original_data=data[int(n * 0.8):, :-1])
    print("\n" + "=" * 40)


    # ================== 预测对比可视化 ==================
    def plot_comparison(y_train_true, train_pred, y_val_true, val_pred):
        """训练集与验证集预测对比"""
        plt.figure(figsize=(14, 6))

        # 训练集子图
        plt.subplot(1, 2, 1)
        plt.scatter(y_train_true, train_pred, c='#1f77b4', alpha=0.6, s=40)
        plt.plot([min(y_train_true), max(y_train_true)],
                 [min(y_train_true), max(y_train_true)],
                 'r--', lw=1.5)
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predictions', fontsize=12)
        plt.title('Training Set Performance', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 验证集子图
        plt.subplot(1, 2, 2)
        plt.scatter(y_val_true, val_pred, c='#ff7f0e', alpha=0.6, s=40)
        plt.plot([min(y_val_true), max(y_val_true)],
                 [min(y_val_true), max(y_val_true)],
                 'r--', lw=1.5)
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predictions', fontsize=12)
        plt.title('Validation Set Performance', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('prediction_comparison.png', dpi=300)
        plt.show()


    # ================== XGBoost特有可视化：特征重要性 ==================
    def plot_feature_importance(model, feature_names):
        """可视化特征重要性"""
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, height=0.8, color='#00A1FF')
        plt.title('Feature Importance', fontsize=14)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.show()


    # ================== 贝叶斯优化结果可视化 ==================
    def plot_optimization_results(result):
        """可视化贝叶斯优化过程"""
        plt.figure(figsize=(12, 6))
        plt.plot(result.func_vals, 'o-', markersize=4)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Cross-Validation RMSE', fontsize=12)
        plt.title('Bayesian Optimization of XGBoost Parameters', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('bayesian_optimization.png', dpi=300)
        plt.show()


    # 获取反归一化后的真实值
    y_train_true = denormalize(y, norms, is_output=True).flatten()
    y_val_true = denormalize(y_val, norms, is_output=True).flatten()

    # 生成可视化图表
    plot_comparison(y_train_true, train_pred, y_val_true, val_pred)

    # 新增：特征重要性可视化
    feature_names = [f'Feature_{i + 1}' for i in range(x.shape[1])]
    plot_feature_importance(best_model, feature_names)

    plot_optimization_results(result)