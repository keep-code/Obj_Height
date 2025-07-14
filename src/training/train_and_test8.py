import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ObstacleHeightClassifier:
    """障碍物高度分类器 - 优化版本"""

    def __init__(self, config=None):
        """
        初始化分类器

        Parameters:
        -----------
        config : dict, optional
            配置字典，包含路径设置等参数
        """
        # 默认配置
        self.config = {
            'model_dir': './models',
            'plot_dir': './plots',
            'results_dir': './results',
            'auto_create_dirs': True,
            'encoding': 'utf-8-sig',  # 解决中文乱码
            'test_size': 0.3,
            'random_state': 42,
            'save_model': True
        }

        # 更新用户配置
        if config:
            self.config.update(config)

        # 模型相关属性
        self.model = None
        self.feature_names = []
        self.test_results = {}

        # 创建输出目录
        if self.config['auto_create_dirs']:
            self._create_directories()

    def _create_directories(self):
        """创建输出目录"""
        for dir_path in [self.config['model_dir'], self.config['plot_dir'], self.config['results_dir']]:
            os.makedirs(dir_path, exist_ok=True)

    def _save_csv(self, df, filepath, encoding=None):
        """保存CSV文件，解决中文乱码问题"""
        encoding = encoding or self.config['encoding']
        try:
            df.to_csv(filepath, index=False, encoding=encoding)
            print(f"文件已保存到: {filepath}")
        except Exception as e:
            print(f"保存文件失败: {e}")
            # 尝试其他编码
            try:
                df.to_csv(filepath, index=False, encoding='gbk')
                print(f"使用GBK编码保存到: {filepath}")
            except:
                df.to_csv(filepath, index=False, encoding='utf-8')
                print(f"使用UTF-8编码保存到: {filepath}")

    def load_data(self, filepath):
        """加载数据"""
        print(f"加载数据: {filepath}")
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        except:
            try:
                df = pd.read_csv(filepath, encoding='gbk')
            except:
                df = pd.read_csv(filepath)

        print(f"数据形状: {df.shape}")
        if 'HeightLabel' in df.columns:
            print(f"HeightLabel分布: {df['HeightLabel'].value_counts().to_dict()}")
            print(f"高障碍物比例: {df['HeightLabel'].mean():.2%}")

        return df

    def feature_engineering(self, df):
        """特征工程"""
        print("进行特征工程...")
        df_processed = df.copy()

        # 排除非特征列
        exclude_cols = ['HeightLabel', 'Train_OD_Project', 'ObjName', 'Direction']

        # 创建新特征
        feature_ops = {
            # 比值特征
            'DeEcho_Ratio': lambda x: x['PosDeDis1'] / (x['PosDeDis2'] + 1e-8),
            'CeEcho_Ratio': lambda x: x['PosCeDis1'] / (x['PosCeDis2'] + 1e-8),
            'DeAmp_Ratio': lambda x: x['PosDeAmp1'] / (x['PosDeAmp2'] + 1e-8),
            'CeAmp_Ratio': lambda x: x['PosCeAmp1'] / (x['PosCeAmp2'] + 1e-8),

            # 总和特征
            'Total_DeEcho': lambda x: x['PosDeDis1'] + x['PosDeDis2'],
            'Total_CeEcho': lambda x: x['PosCeDis1'] + x['PosCeDis2'],
            'Total_DeAmp': lambda x: x['PosDeAmp1'] + x['PosDeAmp2'],
            'Total_CeAmp': lambda x: x['PosCeAmp1'] + x['PosCeAmp2'],

            # 差值特征
            'DeDis_Diff': lambda x: x['PosDeDis1'] - x['PosDeDis2'],
            'CeDis_Diff': lambda x: x['PosCeDis1'] - x['PosCeDis2'],
            'DeAmp_Diff': lambda x: x['PosDeAmp1'] - x['PosDeAmp2'],
            'CeAmp_Diff': lambda x: x['PosCeAmp1'] - x['PosCeAmp2'],

            # 其他特征
            'Avg_Echo_Strength': lambda x: (x['AvgDeEchoHigh_SameTx'] + x['AvgCeEchoHigh_SameTxRx']) / 2,
            'Distance_Ratio': lambda x: x['TrainObjDist'] / (x['AngleDist'] + 1e-8)
        }

        # 应用特征工程
        for feature_name, operation in feature_ops.items():
            df_processed[feature_name] = operation(df_processed)

        # 更新特征列表
        self.feature_names = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"特征工程完成，共 {len(self.feature_names)} 个特征")

        return df_processed

    def train(self, train_df):
        """训练模型"""
        print("开始训练模型...")

        # 准备数据
        X = train_df[self.feature_names]
        y = train_df['HeightLabel']

        # 划分数据集
        if self.config['test_size'] > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'],
                random_state=self.config['random_state'], stratify=y
            )
            print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
        else:
            X_train, y_train = X, y
            X_test = y_test = None
            print(f"使用全部数据训练: {X_train.shape[0]}")

        # LightGBM参数
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
            'random_state': self.config['random_state'],
            'is_unbalance': True
        }

        # 训练
        train_data = lgb.Dataset(X_train, label=y_train)
        if X_test is not None:
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            valid_sets = [train_data, valid_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']

        self.model = lgb.train(
            params, train_data, valid_sets=valid_sets, valid_names=valid_names,
            num_boost_round=1000, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        # 保存测试结果
        if X_test is not None:
            y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)

            self.test_results = {
                'X_test': X_test, 'y_test': y_test,
                'y_pred': y_pred, 'y_pred_proba': y_pred_proba
            }

            # 评估
            self._evaluate_model()

        # 保存模型
        if self.config['save_model']:
            self.save_model()

        return self.test_results if X_test is not None else None

    def _evaluate_model(self):
        """评估模型性能"""
        if not self.test_results:
            return

        y_test = self.test_results['y_test']
        y_pred = self.test_results['y_pred']
        y_pred_proba = self.test_results['y_pred_proba']

        print("\n=== 模型评估结果 ===")
        print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['低障碍物', '高障碍物']))

        # 绘制混淆矩阵
        self.plot_confusion_matrix()

        # 保存错误分类样本
        self._save_misclassified_samples()

    def plot_confusion_matrix(self, title_suffix=""):
        """绘制混淆矩阵"""
        if not self.test_results:
            print("无测试数据，跳过混淆矩阵绘制")
            return

        y_test = self.test_results['y_test']
        y_pred = self.test_results['y_pred']
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['低障碍物', '高障碍物'],
                    yticklabels=['低障碍物', '高障碍物'])
        plt.title(f'混淆矩阵{title_suffix}')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        # 保存图片
        save_path = os.path.join(self.config['plot_dir'], f'confusion_matrix{title_suffix.replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
        plt.show()

        return cm

    def plot_feature_importance(self, top_n=20):
        """绘制特征重要性"""
        if self.model is None:
            print("模型未训练")
            return

        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp.head(top_n), x='importance', y='feature')
        plt.title(f'前{top_n}个重要特征')
        plt.xlabel('重要性')

        save_path = os.path.join(self.config['plot_dir'], f'feature_importance_top{top_n}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
        plt.show()

        return feature_imp

    def _save_misclassified_samples(self):
        """保存错误分类样本"""
        if not self.test_results:
            return

        X_test = self.test_results['X_test']
        y_test = self.test_results['y_test']
        y_pred = self.test_results['y_pred']
        y_pred_proba = self.test_results['y_pred_proba']

        # 找出错误样本
        misclassified_mask = (y_test != y_pred)
        if not any(misclassified_mask):
            print("所有样本预测正确")
            return

        # 创建错误样本DataFrame
        misclassified_data = []
        misclassified_indices = X_test[misclassified_mask].index

        for idx in misclassified_indices:
            row = {
                'Index': idx,
                'True_Label': y_test[idx],
                'Predicted_Label': y_pred[misclassified_mask][misclassified_indices.get_loc(idx)],
                'Prediction_Probability': y_pred_proba[misclassified_mask][misclassified_indices.get_loc(idx)],
                'Error_Type': 'False_Positive' if y_test[idx] == 0 else 'False_Negative'
            }
            misclassified_data.append(row)

        misclassified_df = pd.DataFrame(misclassified_data)

        # 保存
        save_path = os.path.join(self.config['results_dir'], 'misclassified_samples.csv')
        self._save_csv(misclassified_df, save_path)

        print(f"错误分类样本数: {len(misclassified_df)}")
        print(f"假阳性: {len(misclassified_df[misclassified_df['Error_Type'] == 'False_Positive'])}")
        print(f"假阴性: {len(misclassified_df[misclassified_df['Error_Type'] == 'False_Negative'])}")

    def predict(self, test_filepath, output_filepath=None):
        """预测测试数据"""
        if self.model is None:
            print("模型未训练")
            return None

        print(f"预测测试数据: {test_filepath}")

        # 加载测试数据
        test_df = self.load_data(test_filepath)
        test_df_processed = self.feature_engineering(test_df)

        # 预测
        X_test = test_df_processed[self.feature_names]
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 创建结果
        results = test_df.copy()

        # 添加预测结果
        if 'HeightLabel' in test_df.columns:
            results['True_Label'] = test_df['HeightLabel']

        results['Predicted_HeightLabel'] = y_pred
        results['Prediction_Probability'] = y_pred_proba
        results['Confidence'] = np.abs(y_pred_proba - 0.5) * 2

        if 'HeightLabel' in test_df.columns:
            results['Prediction_Correct'] = (results['True_Label'] == results['Predicted_HeightLabel'])
            results['Error_Type'] = results.apply(
                lambda row: 'Correct' if row['Prediction_Correct'] else
                ('False_Positive' if row['True_Label'] == 0 else 'False_Negative'), axis=1
            )

        # 保存结果
        if output_filepath is None:
            filename = f"prediction_results_{os.path.splitext(os.path.basename(test_filepath))[0]}.csv"
            output_filepath = os.path.join(self.config['results_dir'], filename)

        self._save_csv(results, output_filepath)

        # 统计信息
        print(f"\n预测统计:")
        print(f"预测为高障碍物比例: {y_pred.mean():.2%}")
        print(f"平均预测概率: {y_pred_proba.mean():.4f}")

        # 如果有真实标签，进行评估
        if 'HeightLabel' in test_df.columns:
            y_true = test_df['HeightLabel'].values
            accuracy = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred_proba)

            print(f"测试集准确率: {accuracy:.4f}")
            print(f"测试集AUC: {auc:.4f}")

            # 绘制混淆矩阵
            test_name = os.path.splitext(os.path.basename(test_filepath))[0]
            self.test_results = {
                'y_test': y_true, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba
            }
            self.plot_confusion_matrix(f"_{test_name}")

        return results

    def save_model(self, filepath=None):
        """保存模型"""
        if self.model is None:
            print("模型未训练")
            return

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.config['model_dir'], f'obstacle_classifier_{timestamp}.joblib')

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")

        # 保存特征列表
        feature_file = filepath.replace('.joblib', '_features.txt')
        with open(feature_file, 'w', encoding='utf-8') as f:
            f.write("模型特征列表:\n")
            for i, feature in enumerate(self.feature_names, 1):
                f.write(f"{i}. {feature}\n")

        return filepath

    def load_model(self, filepath):
        """加载模型"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            if 'config' in model_data:
                self.config.update(model_data['config'])

            print(f"模型加载成功: {filepath}")
            print(f"训练时间: {model_data.get('timestamp', '未知')}")
            print(f"特征数量: {len(self.feature_names)}")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False


def quick_run(train_file, test_file=None, config=None):
    """快速运行函数"""
    print("=== 障碍物高度分类器 - 快速运行 ===\n")

    # 创建分类器
    classifier = ObstacleHeightClassifier(config)

    # 训练
    print("1. 训练阶段")
    train_df = classifier.load_data(train_file)
    train_df_processed = classifier.feature_engineering(train_df)
    classifier.train(train_df_processed)

    # 绘制特征重要性
    print("\n2. 特征重要性分析")
    classifier.plot_feature_importance()

    # 预测
    if test_file and os.path.exists(test_file):
        print("\n3. 预测阶段")
        results = classifier.predict(test_file)
    else:
        print("\n3. 跳过预测阶段（无测试文件）")

    print("\n=== 运行完成 ===")
    return classifier


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 在这里修改你的路径和参数设置

    # 文件路径配置
    TRAIN_FILE = r'D:\PythonProject\data\processed_data\train_group1.csv'
    TEST_FILE = r'D:\PythonProject\data\processed_data\train_group2.csv'

    # 输出路径配置
    OUTPUT_CONFIG = {
        'model_dir': r'D:\PythonProject\model\saved_model',
        'plot_dir': r'D:\PythonProject\results\visualization_results',
        'results_dir': r'D:\PythonProject\results\prediction_results',
        'auto_create_dirs': True,
        'encoding': 'utf-8-sig',  # 解决中文乱码问题

        # 训练参数
        'test_size': 0.0,  # 0.0表示使用全部数据训练，0.3表示30%用于测试
        'random_state': 42,
        'save_model': False  # 是否保存模型
    }

    # ==================== 运行区域 ====================

    # 方式1: 快速运行（推荐）
    classifier = quick_run(TRAIN_FILE, TEST_FILE, OUTPUT_CONFIG)

    # 方式2: 分步运行（需要更多控制时使用）
    # classifier = ObstacleHeightClassifier(OUTPUT_CONFIG)
    #
    # # 训练
    # train_df = classifier.load_data(TRAIN_FILE)
    # train_df_processed = classifier.feature_engineering(train_df)
    # classifier.train(train_df_processed)
    #
    # # 特征重要性
    # classifier.plot_feature_importance(top_n=15)
    #
    # # 预测
    # if os.path.exists(TEST_FILE):
    #     results = classifier.predict(TEST_FILE)

    print(f"\n输出目录:")
    print(f"- 模型保存: {OUTPUT_CONFIG['model_dir']}")
    print(f"- 图片保存: {OUTPUT_CONFIG['plot_dir']}")
    print(f"- 结果保存: {OUTPUT_CONFIG['results_dir']}")