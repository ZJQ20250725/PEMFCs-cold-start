"""
Created on 2025.06.01

@Source code author: Zhu, J.Q

This code has been modified to generate an artificial neural network agent model
to simulate the maximum ice formation within the cathode CL during the cold start of PEMFCs.

"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # 新增早停和模型保存回调
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import shap

# 设置全局字体和字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['legend.fontsize'] = 24

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


# ================== 自动化预处理函数 ==================
def auto_normalize(data):
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
    if is_output:
        return (data - norms['output']['min']) / (norms['output']['max'] - norms['output']['min'])
    else:
        return (data - norms['features']['min']) / (norms['features']['max'] - norms['features']['min'])


def denormalize(data, norms, is_output=False):
    if is_output:
        return data * (norms['output']['max'] - norms['output']['min']) + norms['output']['min']
    else:
        return data * (norms['features']['max'] - norms['features']['min']) + norms['features']['min']


# ================== 数据预处理 ==================
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    return x, y


# ================== 残差分析函数 ==================
def analyze_residuals(y_true, predictions, title):
    residuals = y_true - predictions

    # 1. 残差-预测值散点图
    plt.figure(figsize=(12, 8))
    plt.scatter(predictions, residuals, alpha=0.6, edgecolors='w', s=40)
    plt.axhline(0, color='r', linestyle='--', lw=1)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{title} - Residuals vs Predicted')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 2. 残差分布直方图
    plt.figure(figsize=(12, 8))
    sns.histplot(residuals, kde=True, color='#2ca02c')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title(f'{title} - Residual Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 3. Q-Q图
    plt.figure(figsize=(12, 8))
    sm.qqplot(residuals, line='45', fit=True)
    plt.title(f'{title} - Q-Q Plot')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ================== SHAP分析模块 ==================
def shap_analysis(model, x_data, feature_names, title):
    try:
        print("\n开始SHAP分析...")

        def model_predict(data):
            data = np.array(data).astype(np.float32)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            return model.predict(data)

        sample_size = min(120, x_data.shape[0])
        if x_data.shape[0] > sample_size:
            n_quantiles = 5
            quantiles = np.linspace(0, 1, n_quantiles)
            feature_quantiles = {}

            for i in range(x_data.shape[1]):
                feature_quantiles[i] = np.quantile(x_data[:, i], quantiles)

            grid_points = []
            for i in range(n_quantiles):
                for j in range(n_quantiles):
                    if i < x_data.shape[0] and j < x_data.shape[0]:
                        grid_points.append((i, j))

            selected_indices = set()
            for i, j in grid_points:
                mask = (x_data[:, 0] >= feature_quantiles[0][i]) & (
                            x_data[:, 0] < feature_quantiles[0][i + 1]) if i < n_quantiles - 1 else (
                            x_data[:, 0] >= feature_quantiles[0][i])
                if j < x_data.shape[1]:
                    mask = mask & ((x_data[:, 1] >= feature_quantiles[1][j]) & (
                                x_data[:, 1] < feature_quantiles[1][j + 1]) if j < n_quantiles - 1 else (
                                x_data[:, 1] >= feature_quantiles[1][j]))

                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    selected_indices.add(np.random.choice(valid_indices))

            if len(selected_indices) < sample_size:
                remaining = sample_size - len(selected_indices)
                additional_indices = np.random.choice(
                    [i for i in range(x_data.shape[0]) if i not in selected_indices],
                    remaining,
                    replace=False
                )
                selected_indices.update(additional_indices)

            background = x_data[list(selected_indices)]
        else:
            background = x_data

        explainer = shap.KernelExplainer(model_predict, background)
        x_sample_size = min(120, x_data.shape[0])
        x_sample = x_data[:x_sample_size]
        shap_values = explainer.shap_values(x_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        expected_value = np.mean(model_predict(background))
        shap_explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(len(shap_values), expected_value),
            data=x_sample,
            feature_names=feature_names
        )

        plt.figure(figsize=(14, 8))
        shap.plots.waterfall(shap_explanation[0], max_display=10, show=False)
        plt.title(f"{title} - SHAP Waterfall Plot for Sample 0")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 8))
        shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
        plt.title(f"{title} - SHAP Beeswarm Plot")
        plt.tight_layout()
        plt.show()

        print("SHAP分析完成，图表已显示。")

    except Exception as e:
        print(f"SHAP分析过程中出现错误: {str(e)}")
        print("建议尝试以下解决方案:")
        print("1. 升级SHAP库到最新版本: pip install --upgrade shap")
        print("2. 减少样本量，或使用更简单的解释器 (如TreeExplainer)")
        print("3. 确保模型输入数据类型与模型期望一致")


# ================== 主程序 ==================
if __name__ == "__main__":
    # 数据加载与处理
    data = pd.read_csv("TRAIN_icevolume.csv")
    data = np.array(data)

    norms = auto_normalize(data)
    xa = apply_normalization(data[:, :-1], norms)
    ya = apply_normalization(data[:, [-1]], norms, is_output=True)

    permutation = np.random.permutation(xa.shape[0])
    xa = xa[permutation, :]
    ya = ya[permutation, :]

    n = xa.shape[0]
    x, x_val = tf.split(xa, num_or_size_splits=[int(n * 0.8), -1])
    y, y_val = tf.split(ya, num_or_size_splits=[int(n * 0.8), -1])

    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(x.shape[0]).map(preprocess).batch(100)

    val_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_db = val_db.map(preprocess).batch(100)

    # ================== 模型定义 (增加正则化) ==================
    model = Sequential([
        # 增加L2正则化和Dropout层
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.05),  # 随机失活5%的神经元
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.05),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.05),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.05),
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1, activation='sigmoid')
    ])

    model.build(input_shape=[None, xa.shape[1]])
    model.summary()

    # ================== 模型训练 (增加早停策略) ==================
    # 早停策略：当验证集损失连续50轮不下降时停止训练
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证集损失
        patience=50,         # 容忍50轮不提升
        restore_best_weights=True  # 恢复最佳权重
    )

    # 保存训练过程中最好的模型
    checkpoint = ModelCheckpoint(
        'best_icevolume_ANN.h5',
        monitor='val_loss',
        save_best_only=True
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=tf.losses.Huber(),
        metrics=['MAPE']
    )

    # 加入早停和模型保存回调
    history = model.fit(
        train_db,
        epochs=4000,
        validation_data=val_db,
        validation_freq=1,  # 每轮都验证（更及时监控过拟合）
        callbacks=[early_stopping, checkpoint]
    )

    # 绘制训练过程曲线
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')  # 改为每轮验证
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制MAPE曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['MAPE'], label='Training MAPE')
    plt.plot(history.history['val_MAPE'], label='Validation MAPE')  # 改为每轮验证
    plt.xlabel('Epochs')
    plt.ylabel('MAPE')
    plt.title('Training and Validation MAPE')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 加载最佳模型
    model = keras.models.load_model('best_icevolume_ANN.h5')
    model.save('Icevolume_ANN_modified.h5')  # 覆盖原保存路径


    # ================== 结果评估 ==================
    def compute_metrics(y_true, predictions):
        mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
        rmse = np.sqrt(np.mean((y_true - predictions) **2))
        r = np.corrcoef(y_true, predictions)[0, 1]
        r2 = r** 2
        return mape, rmse, r, r2


    def evaluate_and_analyze(x_data, y_data, title, original_data=None):
        py = model(x_data)

        y_true = denormalize(y_data.numpy(), norms, is_output=True).flatten()
        predictions = denormalize(py.numpy(), norms, is_output=True).flatten()

        mape, rmse, r, r2 = compute_metrics(y_true, predictions)

        print(f"\n{title} Results:")
        print(f"MAPE: {mape:.2f}%")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Pearson R: {r:.4f}")

        analyze_residuals(y_true, predictions, title)

        if original_data is not None:
            x_denorm = denormalize(x_data.numpy(), norms)
            df = pd.DataFrame(x_denorm, columns=[f'Feature_{i + 1}' for i in range(x_denorm.shape[1])])
            df['True_Output'] = y_true
            df['Predicted_Output'] = predictions
            filename = f"{title.lower().replace(' ', '_')}_results.csv"
            df.to_csv(filename, index=False)
            print(f"\nSaved {filename} with {len(df)} records.")

        return predictions


    print("\n" + "=" * 40)
    _ = evaluate_and_analyze(x, y, "Training Set", original_data=data[:int(n * 0.8), :-1])
    print("\n" + "=" * 40)
    _ = evaluate_and_analyze(x_val, y_val, "Validation Set", original_data=data[int(n * 0.8):, :-1])
    print("\n" + "=" * 40)

    # ================== SHAP分析 ==================
    x_norm = x.numpy()
    feature_names = [f'Feature_{i + 1}' for i in range(x.shape[1])]
    shap_analysis(model, x_norm[:200], feature_names, "Training Set")


    # ================== 预测对比可视化 ==================
    def plot_comparison(y_train_true, train_pred, y_val_true, val_pred):
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(y_train_true, train_pred, c='#1f77b4', alpha=0.6, s=40)
        plt.plot([min(y_train_true), max(y_train_true)],
                 [min(y_train_true), max(y_train_true)],
                 'r--', lw=1.5)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Training Set Performance')
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.scatter(y_val_true, val_pred, c='#ff7f0e', alpha=0.6, s=40)
        plt.plot([min(y_val_true), max(y_val_true)],
                 [min(y_val_true), max(y_val_true)],
                 'r--', lw=1.5)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Validation Set Performance')
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()


    train_pred = denormalize(model.predict(x), norms, is_output=True).flatten()
    val_pred = denormalize(model.predict(x_val), norms, is_output=True).flatten()
    y_train_true = denormalize(y.numpy(), norms, is_output=True).flatten()
    y_val_true = denormalize(y_val.numpy(), norms, is_output=True).flatten()

    plot_comparison(y_train_true, train_pred, y_val_true, val_pred)