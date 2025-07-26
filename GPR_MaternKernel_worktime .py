"""
Created on 2025.06.06

@Source code author: Zhu, J.Q

This code has been modified to generate an optimized GPR proxy model with refined
Bayesian hyperparameter tuning for simulating the time required for a PEMFCs to warm up from 30℃ to 70℃ during cold-start.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C, WhiteKernel
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
import joblib
from sklearn.model_selection import KFold

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


# ================== 贝叶斯优化GPR核参数 ==================
def objective(params):
    """
    贝叶斯优化的目标函数：最小化交叉验证的RMSE
    """
    kernel_type, length_scale, constant_value, noise_level = params

    # 构建核函数
    if kernel_type == 'rbf':
        base_kernel = C(constant_value, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
    elif kernel_type == 'matern':
        base_kernel = C(constant_value, (1e-3, 1e3)) * Matern(length_scale, length_scale_bounds=(1e-2, 1e2))
    elif kernel_type == 'rq':
        base_kernel = C(constant_value, (1e-3, 1e3)) * RationalQuadratic(length_scale, alpha_bounds=(1e-2, 1e2))
    else:
        raise ValueError("Unsupported kernel type")

    full_kernel = base_kernel + WhiteKernel(noise_level, noise_level_bounds=(1e-10, 1e+1))

    # 初始化GPR模型
    model = GaussianProcessRegressor(
        kernel=full_kernel,
        n_restarts_optimizer=0,  # 由贝叶斯优化替代
        alpha=0.0,
        normalize_y=False,
        random_state=2345
    )

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=2345)
    rmse_scores = []

    for train_idx, val_idx in kf.split(xa):
        x_train, x_val = xa[train_idx], xa[val_idx]
        y_train, y_val = ya[train_idx].flatten(), ya[val_idx].flatten()

        model.fit(x_train, y_train)
        y_pred, _ = model.predict(x_val, return_std=True)

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

    # ================== 贝叶斯优化GPR核参数 ==================
    print("\n开始贝叶斯优化GPR核参数（优化搜索边界）...")

    # 定义优化后的搜索空间（边界调整）
    search_space = [
        Categorical(['rbf', 'matern', 'rq'], name='kernel_type'),  # 增加有理二次核
        Real(0.5, 5.0, name='length_scale'),  # 缩小长度尺度范围（原0.1-10 → 0.5-5）
        Real(0.5, 5.0, name='constant_value'),  # 缩小常数核振幅范围
        Real(1e-4, 0.1, name='noise_level'),  # 细化噪声搜索区间
    ]

    # 执行优化（增加迭代次数）
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=70,  # 增加迭代次数从50→70
        n_random_starts=15,  # 增加随机起点从10→15
        random_state=2345,
        verbose=True
    )

    # 输出最优参数
    print("\n最优核参数:")
    print(f"核类型: {result.x[0]}")
    print(f"长度尺度: {result.x[1]:.4f}")
    print(f"常数核振幅: {result.x[2]:.4f}")
    print(f"噪声水平: {result.x[3]:.8f}")
    print(f"交叉验证RMSE: {result.fun:.6f}")

    # ================== 使用最优参数训练最终模型 ==================
    print("\n使用最优参数训练最终GPR模型...")

    # 构建最优核
    if result.x[0] == 'rbf':
        best_kernel = C(result.x[2], (1e-3, 1e3)) * RBF(result.x[1], (1e-2, 1e2))
    elif result.x[0] == 'matern':
        best_kernel = C(result.x[2], (1e-3, 1e3)) * Matern(result.x[1], length_scale_bounds=(1e-2, 1e2))
    else:  # rq
        best_kernel = C(result.x[2], (1e-3, 1e3)) * RationalQuadratic(result.x[1], alpha_bounds=(1e-2, 1e2))

    best_kernel += WhiteKernel(result.x[3], noise_level_bounds=(1e-10, 1e+1))

    # 训练最终模型
    model = GaussianProcessRegressor(
        kernel=best_kernel,
        n_restarts_optimizer=10,
        alpha=0.0,
        normalize_y=False,
        random_state=2345
    )

    model.fit(x, y)

    # 保存模型
    joblib.dump(model, 'worktime_GPR_optimized_bounded.pkl')
    print(f"优化后的GPR模型已保存: worktime_GPR_optimized_bounded.pkl")
    print(f"优化后的核: {model.kernel_}")


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
        # 获取预测结果和标准差
        predictions, std_dev = model.predict(x_data, return_std=True)

        # 反归一化
        y_true = denormalize(y_data, norms, is_output=True).flatten()
        predictions = denormalize(predictions, norms, is_output=True).flatten()
        std_dev = denormalize(std_dev.reshape(-1, 1), norms, is_output=True).flatten()

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
            df['Prediction_StdDev'] = std_dev

            # 保存到CSV文件
            filename = f"{title.lower().replace(' ', '_')}_results.csv"
            df.to_csv(filename, index=False)
            print(f"\nSaved {filename} with {len(df)} records.")

        return predictions, std_dev


    print("\n" + "=" * 40)
    train_pred, train_std = evaluate_and_analyze(x, y, "Training Set", original_data=data[:int(n * 0.8), :-1])
    print("\n" + "=" * 40)
    val_pred, val_std = evaluate_and_analyze(x_val, y_val, "Validation Set", original_data=data[int(n * 0.8):, :-1])
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


    # ================== GPR特有可视化：预测不确定性 ==================
    def plot_uncertainty(x_data, y_true, predictions, std_dev, title):
        """可视化GPR预测的不确定性"""
        # 只选择前100个样本进行可视化，避免过于拥挤
        if len(x_data) > 100:
            indices = np.random.choice(len(x_data), 100, replace=False)
            x_plot = x_data[indices]
            y_plot = y_true[indices]
            pred_plot = predictions[indices]
            std_plot = std_dev[indices]
        else:
            x_plot = x_data
            y_plot = y_true
            pred_plot = predictions
            std_plot = std_dev

        # 创建排序索引以便于可视化
        sort_idx = np.argsort(y_plot)

        plt.figure(figsize=(12, 6))
        plt.plot(y_plot[sort_idx], 'b-', label='True Values')
        plt.plot(pred_plot[sort_idx], 'r--', label='Predictions')
        plt.fill_between(
            range(len(pred_plot)),
            (pred_plot - 2 * std_plot)[sort_idx],
            (pred_plot + 2 * std_plot)[sort_idx],
            color='r', alpha=0.2, label='95% Confidence Interval'
        )
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Work time', fontsize=12)
        plt.title(f'{title} - Predictions with Uncertainty', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{title}_uncertainty.png', dpi=300)
        plt.show()


    # 获取反归一化后的真实值
    y_train_true = denormalize(y, norms, is_output=True).flatten()
    y_val_true = denormalize(y_val, norms, is_output=True).flatten()

    # 生成可视化图表
    plot_comparison(y_train_true, train_pred, y_val_true, val_pred)
    plot_uncertainty(x, y_train_true, train_pred, train_std, "Training Set")
    plot_uncertainty(x_val, y_val_true, val_pred, val_std, "Validation Set")


    # ================== 贝叶斯优化结果可视化 ==================
    def plot_optimization_results(result):
        """可视化贝叶斯优化过程"""
        plt.figure(figsize=(12, 6))
        plt.plot(result.func_vals, 'o-', markersize=4)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Cross-Validation RMSE', fontsize=12)
        plt.title('Bayesian Optimization of GPR Kernel Parameters', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('bayesian_optimization.png', dpi=300)
        plt.show()


    plot_optimization_results(result)