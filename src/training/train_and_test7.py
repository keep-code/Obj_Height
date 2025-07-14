import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
import warnings
import joblib
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class ObstacleHeightClassifier:
    def __init__(self, output_config=None):
        """
        初始化分类器

        Parameters:
        -----------
        output_config : dict, optional
            输出配置字典，包含以下键值:
            - 'model_save_dir': 模型保存目录 (默认: './models')
            - 'plot_save_dir': 图片保存目录 (默认: './plots')
            - 'results_save_dir': 结果文件保存目录 (默认: './results')
            - 'auto_create_dirs': 是否自动创建目录 (默认: True)
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None

        # 设置默认输出配置
        self.output_config = {
            'model_save_dir': './models',
            'plot_save_dir': './plots',
            'results_save_dir': './results',
            'auto_create_dirs': True
        }

        # 更新用户配置
        if output_config:
            self.output_config.update(output_config)

        # 自动创建目录
        if self.output_config['auto_create_dirs']:
            self._create_output_directories()

    def _create_output_directories(self):
        """创建输出目录"""
        for dir_path in [self.output_config['model_save_dir'],
                         self.output_config['plot_save_dir'],
                         self.output_config['results_save_dir']]:
            os.makedirs(dir_path, exist_ok=True)

    def set_output_config(self, **kwargs):
        """
        更新输出配置

        Parameters:
        -----------
        model_save_dir : str, optional
            模型保存目录
        plot_save_dir : str, optional
            图片保存目录
        results_save_dir : str, optional
            结果文件保存目录
        auto_create_dirs : bool, optional
            是否自动创建目录
        """
        self.output_config.update(kwargs)
        if self.output_config['auto_create_dirs']:
            self._create_output_directories()

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

    def save_model(self, model_path=None, save_model=True):
        """
        保存训练好的模型和相关信息

        Parameters:
        -----------
        model_path : str, optional
            模型保存路径，如果为None则使用默认路径
        save_model : bool, optional
            是否保存模型，默认True
        """
        if not save_model:
            print("跳过模型保存")
            return None

        if self.model is None:
            print("模型未训练，无法保存")
            return None

        # 如果未指定路径，生成默认路径
        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"obstacle_height_classifier_{timestamp}.joblib"
            model_path = os.path.join(self.output_config['model_save_dir'], model_filename)

        # 创建保存目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 保存模型和相关信息
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, model_path)
        print(f"模型已保存到: {model_path}")
        print(f"模型包含特征数量: {len(self.feature_names)}")

        # 同时保存特征名称到文本文件
        feature_file = model_path.replace('.joblib', '_features.txt')
        with open(feature_file, 'w', encoding='utf-8') as f:
            f.write("模型特征列表:\n")
            for i, feature in enumerate(self.feature_names, 1):
                f.write(f"{i}. {feature}\n")

        print(f"特征列表已保存到: {feature_file}")

        return model_path

    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.scaler = model_data['scaler']

            print(f"模型已从 {model_path} 加载成功")
            print(f"模型训练时间: {model_data.get('timestamp', '未知')}")
            print(f"模型特征数量: {len(self.feature_names)}")

            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def train_model(self, df, test_size=0.3, random_state=42, save_model=True, model_path=None):
        """
        训练LightGBM模型

        Parameters:
        -----------
        df : DataFrame
            训练数据
        test_size : float
            测试集比例
        random_state : int
            随机种子
        save_model : bool
            是否保存模型
        model_path : str, optional
            模型保存路径
        """
        print("开始训练模型...")

        # 准备特征和目标变量
        X = df[self.feature_names]
        y = df['HeightLabel']

        # 根据test_size决定是否划分数据集
        if test_size > 0:
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            print(f"训练集大小: {X_train.shape[0]}")
            print(f"测试集大小: {X_test.shape[0]}")

            # 创建LightGBM数据集
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            # 训练时使用验证集
            valid_sets = [train_data, valid_data]
            valid_names = ['train', 'valid']

        else:
            # 使用全部数据训练
            X_train = X
            y_train = y
            X_test = None
            y_test = None

            print(f"使用全部数据训练，训练集大小: {X_train.shape[0]}")

            # 创建LightGBM数据集
            train_data = lgb.Dataset(X_train, label=y_train)

            # 训练时只使用训练集
            valid_sets = [train_data]
            valid_names = ['train']

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
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )

        # 保存模型
        if save_model:
            self.save_model(model_path, save_model=True)

        # 如果有测试集，则进行预测和评估
        if X_test is not None:
            y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # 保存测试结果用于后续分析
            self.X_test = X_test
            self.y_test = y_test
            self.y_pred = y_pred
            self.y_pred_proba = y_pred_proba

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

            # 绘制混淆矩阵图
            self.plot_confusion_matrix(y_test, y_pred)

            # 保存分类错误的样本
            self.save_misclassified_samples(df)

            return X_test, y_test, y_pred, y_pred_proba
        else:
            print("\n模型训练完成（使用全部数据，无测试集评估）")

            # 特征重要性
            self.plot_feature_importance()

            return None, None, None, None

    def plot_confusion_matrix(self, y_true, y_pred, title_suffix="", save_path=None):
        """
        绘制混淆矩阵图

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        title_suffix : str
            标题后缀
        save_path : str, optional
            保存路径，如果为None则使用默认路径
        """
        if y_true is None or y_pred is None:
            print("无测试数据，跳过混淆矩阵绘制")
            return

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low Obstacle', 'High Obstacle'],
                    yticklabels=['Low Obstacle', 'High Obstacle'])
        plt.title(f'Confusion Matrix{title_suffix}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # 生成保存路径
        if save_path is None:
            if title_suffix:
                safe_filename = self._clean_filename(title_suffix.strip('() '))
                filename = f'confusion_matrix_{safe_filename}.png'
            else:
                filename = 'confusion_matrix.png'
            save_path = os.path.join(self.output_config['plot_save_dir'], filename)

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图已保存到: {save_path}")
        plt.show()

        # 计算各项指标
        tn, fp, fn, tp = cm.ravel()
        print(f"\n详细混淆矩阵指标{title_suffix}:")
        print(f"真阴性 (TN): {tn}")
        print(f"假阳性 (FP): {fp}")
        print(f"假阴性 (FN): {fn}")
        print(f"真阳性 (TP): {tp}")
        print(f"精确率 (Precision): {tp / (tp + fp):.4f}")
        print(f"召回率 (Recall): {tp / (tp + fn):.4f}")
        print(f"F1分数: {2 * tp / (2 * tp + fp + fn):.4f}")

        return cm

    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        绘制特征重要性

        Parameters:
        -----------
        top_n : int
            显示前N个重要特征
        save_path : str, optional
            保存路径，如果为None则使用默认路径
        """
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
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()

        # 生成保存路径
        if save_path is None:
            filename = f'feature_importance_top{top_n}.png'
            save_path = os.path.join(self.output_config['plot_save_dir'], filename)

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
        plt.show()

        print(f"\nTop {top_n} 重要特征:")
        print(feature_imp.head(top_n))

        return feature_imp

    def _clean_filename(self, filename):
        """清理文件名中的特殊字符"""
        import re
        # 移除或替换特殊字符
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = filename.replace(' ', '_')
        return filename

    def save_misclassified_samples(self, original_df, suffix="", save_path=None):
        """
        保存分类错误的样本

        Parameters:
        -----------
        original_df : DataFrame
            原始数据
        suffix : str
            文件名后缀
        save_path : str, optional
            保存路径
        """
        if self.X_test is None or self.y_test is None or self.y_pred is None:
            print("无测试数据，跳过错误分类样本保存")
            return

        # 找出错误分类的样本
        misclassified_mask = (self.y_test != self.y_pred)
        misclassified_indices = self.X_test[misclassified_mask].index

        # 从原始数据中提取错误分类的样本
        misclassified_samples = original_df.loc[misclassified_indices].copy()

        # 添加预测信息
        misclassified_samples['Predicted_Label'] = self.y_pred[misclassified_mask]
        misclassified_samples['Prediction_Probability'] = self.y_pred_proba[misclassified_mask]
        misclassified_samples['Error_Type'] = misclassified_samples.apply(
            lambda row: 'False_Positive' if row['HeightLabel'] == 0 else 'False_Negative', axis=1
        )

        # 生成保存路径
        if save_path is None:
            filename = f'misclassified_samples{suffix}.csv'
            save_path = os.path.join(self.output_config['results_save_dir'], filename)

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        misclassified_samples.to_csv(save_path, index=False)

        print(f"\n分类错误样本已保存到: {save_path}")
        print(f"错误分类样本数量: {len(misclassified_samples)}")
        print(f"假阳性样本数量: {len(misclassified_samples[misclassified_samples['Error_Type'] == 'False_Positive'])}")
        print(f"假阴性样本数量: {len(misclassified_samples[misclassified_samples['Error_Type'] == 'False_Negative'])}")

        return misclassified_samples

    def analyze_test_predictions(self, test_df, y_true, y_pred, y_pred_proba, test_file_name=""):
        """分析测试集预测结果，显示混淆矩阵和统计信息"""
        print(f"\n========== 测试集预测结果分析 ==========")

        # 1. 基本统计信息
        print(f"测试集样本总数: {len(y_true)}")
        print(f"真实高障碍物样本数: {sum(y_true)}")
        print(f"真实低障碍物样本数: {len(y_true) - sum(y_true)}")
        print(f"预测高障碍物样本数: {sum(y_pred)}")
        print(f"预测低障碍物样本数: {len(y_pred) - sum(y_pred)}")

        # 2. 模型性能指标
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)

        print(f"\n模型性能指标:")
        print(f"准确率: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")

        # 3. 分类报告
        print(f"\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=['低障碍物', '高障碍物']))

        # 4. 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n混淆矩阵:")
        print(cm)

        # 5. 绘制混淆矩阵图
        if test_file_name:
            filename_only = os.path.basename(test_file_name) if test_file_name else ""
            title_suffix = f" ({filename_only})" if filename_only else " (测试集)"
        else:
            title_suffix = " (测试集)"

        self.plot_confusion_matrix(y_true, y_pred, title_suffix)

        # 6. 预测正确和错误的统计
        correct_predictions = (y_true == y_pred)
        incorrect_predictions = ~correct_predictions

        print(f"\n预测正确性统计:")
        print(f"预测正确样本数: {sum(correct_predictions)}")
        print(f"预测错误样本数: {sum(incorrect_predictions)}")
        print(f"预测正确率: {sum(correct_predictions) / len(y_true):.4f}")

        # 7. 按类别统计预测正确性
        true_low_correct = sum((y_true == 0) & (y_pred == 0))
        true_high_correct = sum((y_true == 1) & (y_pred == 1))
        true_low_total = sum(y_true == 0)
        true_high_total = sum(y_true == 1)

        print(f"\n按类别统计:")
        print(f"低障碍物预测正确: {true_low_correct}/{true_low_total} ({true_low_correct / true_low_total:.4f})")
        print(f"高障碍物预测正确: {true_high_correct}/{true_high_total} ({true_high_correct / true_high_total:.4f})")

        # 8. 保存预测错误的样本
        misclassified_samples = self.save_test_misclassified_samples(
            test_df, y_true, y_pred, y_pred_proba, test_file_name
        )

        return cm, misclassified_samples

    def save_test_misclassified_samples(self, test_df, y_true, y_pred, y_pred_proba, test_file_name=""):
        """保存测试集中分类错误的样本"""
        # 找出错误分类的样本
        misclassified_mask = (y_true != y_pred)

        if not any(misclassified_mask):
            print("所有样本都预测正确，无错误分类样本")
            return None

        # 提取错误分类的样本
        misclassified_samples = test_df[misclassified_mask].copy()

        # 添加预测信息
        misclassified_samples['True_Label'] = y_true[misclassified_mask]
        misclassified_samples['Predicted_Label'] = y_pred[misclassified_mask]
        misclassified_samples['Prediction_Probability'] = y_pred_proba[misclassified_mask]
        misclassified_samples['Error_Type'] = misclassified_samples.apply(
            lambda row: 'False_Positive' if row['True_Label'] == 0 else 'False_Negative', axis=1
        )

        # 计算置信度（距离0.5越远置信度越高）
        misclassified_samples['Confidence'] = np.abs(y_pred_proba[misclassified_mask] - 0.5) * 2

        # 生成文件名
        if test_file_name:
            filename_part = self._clean_filename(os.path.splitext(os.path.basename(test_file_name))[0])
            suffix = f"_{filename_part}"
        else:
            suffix = "_test"

        filename = f'misclassified_samples{suffix}.csv'
        save_path = os.path.join(self.output_config['results_save_dir'], filename)

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            misclassified_samples.to_csv(save_path, index=False)
            print(f"\n测试集错误分类样本已保存到: {save_path}")
        except Exception as e:
            print(f"保存错误分类样本时出错: {e}")
            return misclassified_samples

        print(f"错误分类样本数量: {len(misclassified_samples)}")

        false_positive_count = len(misclassified_samples[misclassified_samples['Error_Type'] == 'False_Positive'])
        false_negative_count = len(misclassified_samples[misclassified_samples['Error_Type'] == 'False_Negative'])

        print(f"假阳性样本数量: {false_positive_count}")
        print(f"假阴性样本数量: {false_negative_count}")

        # 显示错误样本的置信度分布
        print(f"\n错误分类样本置信度统计:")
        print(f"平均置信度: {misclassified_samples['Confidence'].mean():.4f}")
        print(f"置信度中位数: {misclassified_samples['Confidence'].median():.4f}")
        print(f"高置信度错误(>0.8): {len(misclassified_samples[misclassified_samples['Confidence'] > 0.8])}")

        return misclassified_samples

    def predict_test_data(self, test_file_path, output_file_path=None):
        """
        预测测试数据并保存结果

        Parameters:
        -----------
        test_file_path : str
            测试文件路径
        output_file_path : str, optional
            输出文件路径，如果为None则使用默认路径
        """
        if self.model is None:
            print("模型未训练，无法进行预测")
            return None

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

        # 添加True_Label列（如果测试数据包含真实标签）
        if 'HeightLabel' in df_test.columns:
            results['True_Label'] = df_test['HeightLabel']

        # 添加预测结果列
        results['Predicted_HeightLabel'] = y_pred
        results['Prediction_Probability'] = y_pred_proba
        results['Confidence'] = np.abs(y_pred_proba - 0.5) * 2  # 置信度：距离0.5越远置信度越高

        # 如果有真实标签，添加预测正确性列
        if 'HeightLabel' in df_test.columns:
            results['Prediction_Correct'] = (results['True_Label'] == results['Predicted_HeightLabel'])
            results['Error_Type'] = results.apply(
                lambda row: 'Correct' if row['Prediction_Correct'] else
                ('False_Positive' if row['True_Label'] == 0 else 'False_Negative'), axis=1
            )

        # 重新排列列的顺序，确保True_Label在Predicted_HeightLabel之前
        if 'HeightLabel' in df_test.columns:
            # 获取原始列
            original_cols = [col for col in df_test.columns if col != 'HeightLabel']

            # 新的列顺序
            new_order = original_cols + ['True_Label', 'Predicted_HeightLabel', 'Prediction_Probability',
                                         'Confidence', 'Prediction_Correct', 'Error_Type']

            results = results[new_order]

        # 生成输出文件路径
        if output_file_path is None:
            test_filename = self._clean_filename(os.path.splitext(os.path.basename(test_file_path))[0])
            filename = f'prediction_results_{test_filename}.csv'
            output_file_path = os.path.join(self.output_config['results_save_dir'], filename)

        # 确保保存目录存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # 保存结果
        results.to_csv(output_file_path, index=False)
        print(f"预测结果已保存到: {output_file_path}")

        # 显示预测统计
        print(f"\n预测统计:")
        print(f"预测为高障碍物的比例: {y_pred.mean():.2%}")
        print(f"平均预测概率: {y_pred_proba.mean():.4f}")
        print(f"平均置信度: {results['Confidence'].mean():.4f}")

        # 如果测试数据包含真实标签，进行详细分析
        if 'HeightLabel' in df_test.columns:
            print(f"\n检测到测试数据包含真实标签，进行详细分析...")
            y_true = df_test['HeightLabel'].values

            # 获取测试文件名用于标识
            test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]

            # 分析预测结果
            cm, misclassified_samples = self.analyze_test_predictions(
                df_test, y_true, y_pred, y_pred_proba, test_file_name
            )

            # 显示预测正确性统计
            correct_count = sum(results['Prediction_Correct'])
            total_count = len(results)
            print(f"\n预测正确性统计:")
            print(f"预测正确样本数: {correct_count}/{total_count} ({correct_count / total_count:.2%})")

            # 按错误类型统计
            error_stats = results['Error_Type'].value_counts()
            print(f"\n错误类型统计:")
            for error_type, count in error_stats.items():
                print(f"{error_type}: {count} ({count / total_count:.2%})")

        return results


# 使用示例
def main():
    """
    使用示例 - 展示如何配置路径和使用增强功能
    """

    # # 方法1: 使用默认配置
    # print("=== 方法1: 使用默认配置 ===")
    # classifier1 = ObstacleHeightClassifier()

    # 方法2: 自定义输出路径配置
    # print("\n=== 方法2: 自定义输出路径配置 ===")
    output_config = {
        'model_save_dir': r'D:\PythonProject\model\saved_model',
        'plot_save_dir': r'D:\PythonProject\results\visualization_results',
        'results_save_dir': r'D:\PythonProject\results\misclassified_results',
        'auto_create_dirs': True    # 自动创建目录
    }
    classifier2 = ObstacleHeightClassifier(output_config=output_config)

    # # 方法3: 运行时修改配置
    # print("\n=== 方法3: 运行时修改配置 ===")
    # classifier3 = ObstacleHeightClassifier()
    # classifier3.set_output_config(
    #     model_save_dir='./custom_models',
    #     plot_save_dir='./custom_plots',
    #     results_save_dir='./custom_results'
    # )

    # 使用配置好的分类器进行训练和预测
    classifier = classifier2  # 使用方法2的配置

    # 1. 加载和预处理训练数据
    train_df = classifier.load_and_preprocess_data(r'D:\PythonProject\data\processed_data\train_group1.csv')

    # 2. 特征工程
    train_df_processed = classifier.feature_engineering(train_df)

    # 3. 训练模型 - 可以指定是否保存模型和保存路径
    X_test, y_test, y_pred, y_pred_proba = classifier.train_model(
        train_df_processed,
        test_size=0.0,
        random_state=42,
        save_model=True,  # 是否保存模型
        model_path=r'D:\PythonProject\model\saved_model\obstacle_height_classifier.joblib'  # 使用默认路径，也可以指定具体路径
    )

    # 4. 预测测试数据
    try:
        test_results = classifier.predict_test_data(
            test_file_path=r'D:\PythonProject\data\processed_data\train_group2.csv',
            output_file_path=r'D:\PythonProject\results\prediction_results\Prediction_Results.csv'  # 使用默认路径，也可以指定具体路径
        )
        print("\n预测完成！")
        print("所有文件已按配置保存到指定目录")
    except FileNotFoundError:
        print("\n测试文件不存在，跳过预测步骤")

    # # 5. 单独绘制图片（可指定保存路径）
    # if classifier.model is not None:
    #     # 绘制特征重要性 - 可指定保存路径
    #     classifier.plot_feature_importance(
    #         top_n=15,
    #         save_path=r'D:\PythonProject\results\visualization_results\custom_feature_importance.png'  # 使用默认路径，也可以指定如: r'D:\PythonProject\plots\custom_feature_importance.png'
    #     )

    print(f"\n输出配置:")
    print(f"模型保存目录: {classifier.output_config['model_save_dir']}")
    print(f"图片保存目录: {classifier.output_config['plot_save_dir']}")
    print(f"结果保存目录: {classifier.output_config['results_save_dir']}")


# 便捷函数 - 用于快速配置和使用
def create_classifier_with_paths(model_dir=None, plot_dir=None, results_dir=None):
    """
    便捷函数：创建配置好路径的分类器

    Parameters:
    -----------
    model_dir : str, optional
        模型保存目录
    plot_dir : str, optional
        图片保存目录
    results_dir : str, optional
        结果保存目录

    Returns:
    --------
    ObstacleHeightClassifier
        配置好的分类器实例
    """
    config = {}
    if model_dir:
        config['model_save_dir'] = model_dir
    if plot_dir:
        config['plot_save_dir'] = plot_dir
    if results_dir:
        config['results_save_dir'] = results_dir

    return ObstacleHeightClassifier(output_config=config)


def quick_train_and_predict(train_file, test_file, output_dir=None, save_model=True):
    """
    快速训练和预测函数

    Parameters:
    -----------
    train_file : str
        训练文件路径
    test_file : str
        测试文件路径
    output_dir : str, optional
        输出目录，如果为None则使用当前目录
    save_model : bool
        是否保存模型
    """
    if output_dir is None:
        output_dir = './output'

    # 创建分类器
    classifier = create_classifier_with_paths(
        model_dir=os.path.join(output_dir, 'models'),
        plot_dir=os.path.join(output_dir, 'plots'),
        results_dir=os.path.join(output_dir, 'results')
    )

    # 训练
    train_df = classifier.load_and_preprocess_data(train_file)
    train_df_processed = classifier.feature_engineering(train_df)
    classifier.train_model(train_df_processed, test_size=0.0, save_model=save_model)

    # 预测
    if os.path.exists(test_file):
        classifier.predict_test_data(test_file)
        print(f"\n所有结果已保存到: {output_dir}")
    else:
        print(f"测试文件不存在: {test_file}")

    return classifier


if __name__ == "__main__":
    main()