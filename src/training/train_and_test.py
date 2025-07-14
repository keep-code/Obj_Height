import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ObstacleHeightClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_and_preprocess_data(self, train_file_path):
        """加载并预处理训练数据"""
        print("加载训练数据...")
        df = pd.read_csv(train_file_path)
        print(f"数据形状: {df.shape}")

        # 显示基本信息
        print("\n数据基本信息:")
        print(df.info())

        # 检查目标变量分布
        print(f"\nHeightLabel分布:")
        print(df['HeightLabel'].value_counts())
        print(f"高障碍物比例: {df['HeightLabel'].mean():.2%}")

        # 检查缺失值
        print(f"\n缺失值统计:")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(missing_data[missing_data > 0])
        else:
            print("无缺失值")

        return df

    def feature_engineering(self, df):
        """特征工程"""
        print("进行特征工程...")

        # 复制数据
        df_processed = df.copy()

        # 选择数值特征（排除目标变量和非数值特征）
        exclude_cols = ['HeightLabel', 'Train_OD_Project', 'ObjName', 'Direction']
        numeric_features = [col for col in df_processed.columns if col not in exclude_cols]

        # 创建新特征
        # 1. 直接回波与间接回波的比值特征
        df_processed['DeEcho_Ratio'] = df_processed['PosDeDis1'] / (df_processed['PosDeDis2'] + 1e-8)
        df_processed['CeEcho_Ratio'] = df_processed['PosCeDis1'] / (df_processed['PosCeDis2'] + 1e-8)

        # 2. 幅值特征
        df_processed['DeAmp_Ratio'] = df_processed['PosDeAmp1'] / (df_processed['PosDeAmp2'] + 1e-8)
        df_processed['CeAmp_Ratio'] = df_processed['PosCeAmp1'] / (df_processed['PosCeAmp2'] + 1e-8)

        # 3. 总体回波强度
        df_processed['Total_DeEcho'] = df_processed['PosDeDis1'] + df_processed['PosDeDis2']
        df_processed['Total_CeEcho'] = df_processed['PosCeDis1'] + df_processed['PosCeDis2']
        df_processed['Total_DeAmp'] = df_processed['PosDeAmp1'] + df_processed['PosDeAmp2']
        df_processed['Total_CeAmp'] = df_processed['PosCeAmp1'] + df_processed['PosCeAmp2']

        # 4. 直接回波与间接回波的差值
        df_processed['DeDis_Diff'] = df_processed['PosDeDis1'] - df_processed['PosDeDis2']
        df_processed['CeDis_Diff'] = df_processed['PosCeDis1'] - df_processed['PosCeDis2']
        df_processed['DeAmp_Diff'] = df_processed['PosDeAmp1'] - df_processed['PosDeAmp2']
        df_processed['CeAmp_Diff'] = df_processed['PosCeAmp1'] - df_processed['PosCeAmp2']

        # 5. 平均回波特征
        df_processed['Avg_Echo_Strength'] = (df_processed['AvgDeEchoHigh_SameTx'] +
                                             df_processed['AvgCeEchoHigh_SameTxRx']) / 2

        # 6. 距离相关特征
        df_processed['Distance_Ratio'] = df_processed['TrainObjDist'] / (df_processed['AngleDist'] + 1e-8)

        # 更新特征列表
        self.feature_names = [col for col in df_processed.columns if col not in exclude_cols]

        print(f"特征工程完成，共生成 {len(self.feature_names)} 个特征")

        return df_processed

    def train_model(self, df, test_size=0.3, random_state=42):
        """训练LightGBM模型"""
        print("开始训练模型...")

        # 准备特征和目标变量
        X = df[self.feature_names]
        y = df['HeightLabel']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")

        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # 设置参数
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state,
            'is_unbalance': True  # 处理类别不平衡
        }

        # 训练模型
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )

        # 预测
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 评估模型
        print("\n模型评估结果:")
        print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

        print("\n分类报告:")
        print(classification_report(y_test, y_pred,
                                    target_names=['低障碍物', '高障碍物']))

        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # 特征重要性
        self.plot_feature_importance()

        return X_test, y_test, y_pred, y_pred_proba

    def plot_feature_importance(self, top_n=20):
        """绘制特征重要性"""
        if self.model is None:
            print("模型未训练，无法显示特征重要性")
            return

        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} 特征重要性')
        plt.xlabel('重要性')
        plt.tight_layout()
        plt.show()

        print(f"\nTop {top_n} 重要特征:")
        print(feature_imp.head(top_n))

    def predict_test_data(self, test_file_path, output_file_path):
        """预测测试数据并保存结果"""
        print(f"加载测试数据: {test_file_path}")

        # 加载测试数据
        df_test = pd.read_csv(test_file_path)
        print(f"测试数据形状: {df_test.shape}")

        # 特征工程（与训练数据相同的处理）
        df_test_processed = self.feature_engineering(df_test)

        # 确保测试数据有相同的特征
        X_test = df_test_processed[self.feature_names]

        # 预测
        print("进行预测...")
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 创建结果DataFrame
        results = df_test.copy()
        results['Predicted_HeightLabel'] = y_pred
        results['Prediction_Probability'] = y_pred_proba
        results['Confidence'] = np.abs(y_pred_proba - 0.5) * 2  # 置信度：距离0.5越远置信度越高

        # 保存结果
        results.to_csv(output_file_path, index=False)
        print(f"预测结果已保存到: {output_file_path}")

        # 显示预测统计
        print(f"\n预测统计:")
        print(f"预测为高障碍物的比例: {y_pred.mean():.2%}")
        print(f"平均预测概率: {y_pred_proba.mean():.4f}")
        print(f"平均置信度: {results['Confidence'].mean():.4f}")

        return results


# 使用示例
def main():
    # 创建分类器实例
    classifier = ObstacleHeightClassifier()

    # 1. 加载和预处理训练数据
    train_df = classifier.load_and_preprocess_data('Train_OD(3).csv')

    # 2. 特征工程
    train_df_processed = classifier.feature_engineering(train_df)

    # 3. 训练模型（70%训练，30%测试）
    X_test, y_test, y_pred, y_pred_proba = classifier.train_model(
        train_df_processed, test_size=0.5, random_state=42
    )

    # 4. 预测测试数据（如果有新的测试文件）
    # 注意：这里假设测试文件名为 'Test_OD.csv'，请根据实际情况修改
    try:
        test_results = classifier.predict_test_data('Test.csv', 'Prediction_Results.csv')
        print("\n预测完成！")
    except FileNotFoundError:
        print("\n测试文件不存在，跳过预测步骤")
        print("如需预测新数据，请将测试数据保存为 'Test_OD.csv' 并重新运行")


if __name__ == "__main__":
    main()