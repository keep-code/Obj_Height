import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, \
    precision_recall_curve
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import logging

warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ObstacleHeightClassifier:
    """障碍物高度分类器 - 改进版本"""

    def __init__(self, config: Optional[Dict] = None):
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
            'misclassified_dir': './misclassified',
            'log_dir': './logs',
            'auto_create_dirs': True,
            'encoding': 'utf-8-sig',
            'test_size': 0.0,
            'random_state': 42,
            'save_model': True,
            'cv_folds': 5,  # 交叉验证折数
            'threshold_optimization': True,  # 是否优化分类阈值
            'feature_selection': True,  # 是否进行特征选择
            'model_params': {  # LightGBM参数
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'is_unbalance': True
            }
        }

        # 更新用户配置
        if config:
            self.config.update(config)
            # 深度更新模型参数
            if 'model_params' in config:
                self.config['model_params'].update(config['model_params'])

        # 模型相关属性
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.selected_features = []
        self.test_results = {}
        self.optimal_threshold = 0.5
        self.cv_scores = {}

        # 设置日志
        self._setup_logging()

        # 创建输出目录
        if self.config['auto_create_dirs']:
            self._create_directories()

    def _setup_logging(self):
        """设置日志记录"""
        log_dir = self.config['log_dir']
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'classifier_{timestamp}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_directories(self):
        """创建输出目录"""
        for dir_path in [self.config['model_dir'], self.config['plot_dir'],
                         self.config['results_dir'], self.config['misclassified_dir'],
                         self.config['log_dir']]:
            os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"确保目录存在: {dir_path}")

    def _save_csv(self, df: pd.DataFrame, filepath: str, encoding: Optional[str] = None) -> bool:
        """保存CSV文件，解决中文乱码问题"""
        encoding = encoding or self.config['encoding']
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False, encoding=encoding)
            self.logger.info(f"文件已保存到: {filepath} ({os.path.getsize(filepath)} bytes)")
            return True
        except Exception as e:
            self.logger.error(f"保存文件失败: {e}")
            # 尝试其他编码
            for alt_encoding in ['gbk', 'utf-8']:
                try:
                    df.to_csv(filepath, index=False, encoding=alt_encoding)
                    self.logger.info(f"使用{alt_encoding}编码保存到: {filepath}")
                    return True
                except Exception:
                    continue
            self.logger.error("所有编码尝试失败")
            return False

    def load_data(self, filepath: str) -> pd.DataFrame:
        """加载数据"""
        self.logger.info(f"加载数据: {filepath}")

        # 尝试不同编码读取
        for encoding in ['utf-8-sig', 'gbk', 'utf-8']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except Exception as e:
                if encoding == 'utf-8':  # 最后一次尝试
                    raise e
                continue

        self.logger.info(f"数据形状: {df.shape}")

        # 数据质量检查
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            self.logger.warning(f"发现缺失值:\n{missing_info[missing_info > 0]}")

        if 'HeightLabel' in df.columns:
            self.logger.info(f"HeightLabel分布: {df['HeightLabel'].value_counts().to_dict()}")
            self.logger.info(f"高障碍物比例: {df['HeightLabel'].mean():.2%}")

            # 检查类别不平衡
            class_ratio = df['HeightLabel'].value_counts()
            minority_ratio = min(class_ratio) / max(class_ratio)
            if minority_ratio < 0.1:
                self.logger.warning(f"类别严重不平衡，少数类比例: {minority_ratio:.2%}")

        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """增强的特征工程"""
        self.logger.info("进行特征工程...")
        df_processed = df.copy()

        # 排除非特征列
        exclude_cols = ['HeightLabel', 'Train_OD_Project', 'ObjName', 'Direction']

        # 基础特征工程
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

            # 组合特征
            'Avg_Echo_Strength': lambda x: (x['AvgDeEchoHigh_SameTx'] + x['AvgCeEchoHigh_SameTxRx']) / 2,
            'Distance_Ratio': lambda x: x['TrainObjDist'] / (x['AngleDist'] + 1e-8),

            # 新增特征
            'Echo_Consistency': lambda x: 1 - abs(x['PosDeDis1'] - x['PosCeDis1']) / (
                        x['PosDeDis1'] + x['PosCeDis1'] + 1e-8),
            'Amp_Consistency': lambda x: 1 - abs(x['PosDeAmp1'] - x['PosCeAmp1']) / (
                        x['PosDeAmp1'] + x['PosCeAmp1'] + 1e-8),
            'Signal_Quality': lambda x: (x['PosDeAmp1'] + x['PosCeAmp1']) / (x['PosDeAmp2'] + x['PosCeAmp2'] + 1e-8)
        }

        # 应用特征工程
        for feature_name, operation in feature_ops.items():
            try:
                df_processed[feature_name] = operation(df_processed)
            except Exception as e:
                self.logger.warning(f"特征 {feature_name} 创建失败: {e}")

        # 处理异常值和无穷值
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)

        # 更新特征列表
        self.feature_names = [col for col in df_processed.columns if col not in exclude_cols]
        self.logger.info(f"特征工程完成，共 {len(self.feature_names)} 个特征")

        # 特征选择
        if self.config['feature_selection'] and 'HeightLabel' in df_processed.columns:
            self._feature_selection(df_processed)

        return df_processed

    def _feature_selection(self, df: pd.DataFrame):
        """基于相关性和重要性的特征选择"""
        X = df[self.feature_names].fillna(0)
        y = df['HeightLabel']

        # 计算特征与目标的相关性
        correlations = []
        for feature in self.feature_names:
            try:
                corr = abs(X[feature].corr(y))
                correlations.append((feature, corr))
            except:
                correlations.append((feature, 0))

        # 按相关性排序，选择前80%的特征
        correlations.sort(key=lambda x: x[1], reverse=True)
        n_select = max(10, int(len(self.feature_names) * 0.8))
        self.selected_features = [feat for feat, _ in correlations[:n_select]]

        self.logger.info(f"特征选择完成，保留 {len(self.selected_features)} 个特征")

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """交叉验证"""
        self.logger.info(f"开始 {self.config['cv_folds']} 折交叉验证...")

        cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True,
                             random_state=self.config['random_state'])

        cv_scores = {'accuracy': [], 'auc': [], 'precision': [], 'recall': []}

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            self.logger.info(f"  训练第 {fold + 1}/{self.config['cv_folds']} 折...")

            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

            # 创建训练和验证数据集
            train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
            val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)

            # 训练模型（使用验证集进行早停）
            model_cv = lgb.train(
                self.config['model_params'],
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            # 预测和评估
            y_pred_proba = model_cv.predict(X_val_cv, num_iteration=model_cv.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)

            from sklearn.metrics import precision_score, recall_score
            cv_scores['accuracy'].append(accuracy_score(y_val_cv, y_pred))
            cv_scores['auc'].append(roc_auc_score(y_val_cv, y_pred_proba))
            cv_scores['precision'].append(precision_score(y_val_cv, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_cv, y_pred, zero_division=0))

        # 计算平均分数和标准差
        self.cv_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        cv_stds = {metric: np.std(scores) for metric, scores in cv_scores.items()}

        self.logger.info("交叉验证结果:")
        for metric, score in self.cv_scores.items():
            self.logger.info(f"  {metric}: {score:.4f} ± {cv_stds[metric]:.4f}")

        return self.cv_scores

    def optimize_threshold(self, X: pd.DataFrame, y: pd.Series):
        """优化分类阈值"""
        if not self.config['threshold_optimization']:
            return

        self.logger.info("优化分类阈值...")

        # 使用训练好的模型预测概率
        # 根据模型是否有best_iteration属性来决定预测方式
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        else:
            y_pred_proba = self.model.predict(X)

        # 计算不同阈值下的精确率和召回率
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)

        # 选择F1分数最高的阈值
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        self.logger.info(f"最优阈值: {self.optimal_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")

    def train(self, train_df: pd.DataFrame) -> Optional[Dict]:
        """训练模型"""
        self.logger.info("开始训练模型...")

        # 保存原始训练数据
        self.train_df_original = train_df.copy()

        # 准备数据
        features_to_use = self.selected_features if self.selected_features else self.feature_names
        X = train_df[features_to_use].fillna(0)
        y = train_df['HeightLabel']

        # 数据标准化（可选）
        if self.config.get('normalize_features', False):
            self.scaler = StandardScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)

        # 交叉验证
        if self.config['cv_folds'] > 1:
            self.cross_validate(X, y)

        # 划分数据集
        if self.config['test_size'] > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'],
                random_state=self.config['random_state'], stratify=y
            )
            self.logger.info(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
        else:
            X_train, y_train = X, y
            X_test = y_test = None
            self.logger.info(f"使用全部数据训练: {X_train.shape[0]}")

        # 训练
        train_data = lgb.Dataset(X_train, label=y_train)
        if X_test is not None:
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            valid_sets = [train_data, valid_data]
            valid_names = ['train', 'valid']
            # 有验证集时使用早停
            callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
        else:
            valid_sets = [train_data]
            valid_names = ['train']
            # 没有验证集时不使用早停，只用日志回调
            callbacks = [lgb.log_evaluation(100)]

        self.model = lgb.train(
            self.config['model_params'], train_data,
            valid_sets=valid_sets, valid_names=valid_names,
            num_boost_round=1000,
            callbacks=callbacks
        )

        # 优化阈值
        self.optimize_threshold(X_train, y_train)

        # 保存测试结果
        if X_test is not None:
            # 根据模型是否有best_iteration属性来决定预测方式
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
                y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
            else:
                y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba > self.optimal_threshold).astype(int)

            self.test_results = {
                'X_test': X_test, 'y_test': y_test,
                'y_pred': y_pred, 'y_pred_proba': y_pred_proba
            }
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

        self.logger.info("\n=== 模型评估结果 ===")
        self.logger.info(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
        self.logger.info(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        self.logger.info(f"使用阈值: {self.optimal_threshold:.4f}")

        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['低障碍物', '高障碍物']))

        # 绘制评估图表
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()

        # 保存错误分类样本
        self._save_misclassified_samples(self.train_df_original)

    def plot_roc_curve(self):
        """绘制ROC曲线"""
        if not self.test_results:
            return

        from sklearn.metrics import roc_curve

        y_test = self.test_results['y_test']
        y_pred_proba = self.test_results['y_pred_proba']

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.config['plot_dir'], 'roc_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"ROC曲线已保存到: {save_path}")
        plt.show()

    def plot_precision_recall_curve(self):
        """绘制精确率-召回率曲线"""
        if not self.test_results:
            return

        y_test = self.test_results['y_test']
        y_pred_proba = self.test_results['y_pred_proba']

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='PR Curve')
        plt.axvline(x=recall[np.argmin(np.abs(thresholds - self.optimal_threshold))],
                    color='red', linestyle='--', label=f'最优阈值 ({self.optimal_threshold:.3f})')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.config['plot_dir'], 'precision_recall_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"PR曲线已保存到: {save_path}")
        plt.show()

    def plot_confusion_matrix(self, title_suffix=""):
        """绘制混淆矩阵"""
        if not self.test_results:
            self.logger.warning("无测试数据，跳过混淆矩阵绘制")
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

        save_path = os.path.join(self.config['plot_dir'], f'confusion_matrix{title_suffix.replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"混淆矩阵已保存到: {save_path}")
        plt.show()

        return cm

    def plot_feature_importance(self, top_n: int = 20):
        """绘制特征重要性"""
        if self.model is None:
            self.logger.warning("模型未训练")
            return

        importance = self.model.feature_importance(importance_type='gain')
        features_to_use = self.selected_features if self.selected_features else self.feature_names

        feature_imp = pd.DataFrame({
            'feature': features_to_use,
            'importance': importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_imp.head(top_n), x='importance', y='feature')
        plt.title(f'前{top_n}个重要特征')
        plt.xlabel('重要性')

        save_path = os.path.join(self.config['plot_dir'], f'feature_importance_top{top_n}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"特征重要性图已保存到: {save_path}")
        plt.show()

        # 保存特征重要性数据
        importance_file = os.path.join(self.config['results_dir'], 'feature_importance.csv')
        self._save_csv(feature_imp, importance_file)

        return feature_imp

    def _save_misclassified_samples(self, train_df: Optional[pd.DataFrame] = None, suffix: str = ""):
        """保存错误分类样本"""
        if not self.test_results:
            self.logger.warning("无测试结果，跳过错误样本保存")
            return

        X_test = self.test_results['X_test']
        y_test = self.test_results['y_test']
        y_pred = self.test_results['y_pred']
        y_pred_proba = self.test_results['y_pred_proba']

        misclassified_mask = (y_test != y_pred)
        if not any(misclassified_mask):
            self.logger.info("所有样本预测正确，无错误分类样本")
            return

        self.logger.info(f"发现 {sum(misclassified_mask)} 个错误分类样本")

        misclassified_indices = X_test[misclassified_mask].index

        if train_df is not None:
            misclassified_samples = train_df.loc[misclassified_indices].copy()
        else:
            misclassified_samples = X_test[misclassified_mask].copy()

        # 添加预测信息
        y_test_mis = y_test[misclassified_mask].reset_index(drop=True)
        y_pred_mis = y_pred[misclassified_mask]
        y_pred_proba_mis = y_pred_proba[misclassified_mask]

        misclassified_samples = misclassified_samples.reset_index(drop=True)
        misclassified_samples['True_Label'] = y_test_mis
        misclassified_samples['Predicted_Label'] = y_pred_mis
        misclassified_samples['Prediction_Probability'] = y_pred_proba_mis
        misclassified_samples['Confidence'] = np.abs(y_pred_proba_mis - 0.5) * 2
        misclassified_samples['Error_Type'] = ['False_Positive' if true_label == 0 else 'False_Negative'
                                               for true_label in y_test_mis]
        misclassified_samples['Used_Threshold'] = self.optimal_threshold

        filename = f'misclassified_samples{suffix}.csv'
        save_path = os.path.join(self.config['misclassified_dir'], filename)

        success = self._save_csv(misclassified_samples, save_path)

        if success:
            false_positive_count = sum(misclassified_samples['Error_Type'] == 'False_Positive')
            false_negative_count = sum(misclassified_samples['Error_Type'] == 'False_Negative')

            self.logger.info(f"错误分类样本数: {len(misclassified_samples)}")
            self.logger.info(f"假阳性样本数: {false_positive_count}")
            self.logger.info(f"假阴性样本数: {false_negative_count}")
            self.logger.info(f"平均预测置信度: {misclassified_samples['Confidence'].mean():.4f}")

        return misclassified_samples

    def predict(self, test_filepath: str, output_filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
        """预测测试数据"""
        if self.model is None:
            self.logger.error("模型未训练")
            return None

        self.logger.info(f"预测测试数据: {test_filepath}")

        # 加载测试数据
        test_df = self.load_data(test_filepath)
        test_df_processed = self.feature_engineering(test_df)

        # 预测
        features_to_use = self.selected_features if self.selected_features else self.feature_names
        X_test = test_df_processed[features_to_use].fillna(0)

        # 应用标准化（如果训练时使用了）
        if self.scaler is not None:
            X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # 根据模型是否有best_iteration属性来决定预测方式
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        else:
            y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > self.optimal_threshold).astype(int)

        # 创建结果
        results = test_df.copy()
        if 'HeightLabel' in test_df.columns:
            results['True_Label'] = test_df['HeightLabel']

        results['Predicted_HeightLabel'] = y_pred
        results['Prediction_Probability'] = y_pred_proba
        results['Confidence'] = np.abs(y_pred_proba - 0.5) * 2
        results['Used_Threshold'] = self.optimal_threshold

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
        self.logger.info(f"\n预测统计:")
        self.logger.info(f"预测为高障碍物比例: {y_pred.mean():.2%}")
        self.logger.info(f"平均预测概率: {y_pred_proba.mean():.4f}")
        self.logger.info(f"使用阈值: {self.optimal_threshold:.4f}")

        # 如果有真实标签，进行评估
        if 'HeightLabel' in test_df.columns:
            y_true = test_df['HeightLabel'].values
            accuracy = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred_proba)

            self.logger.info(f"测试集准确率: {accuracy:.4f}")
            self.logger.info(f"测试集AUC: {auc:.4f}")

            # 绘制混淆矩阵
            test_name = os.path.splitext(os.path.basename(test_filepath))[0]
            self.test_results = {
                'y_test': y_true, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba
            }
            self.plot_confusion_matrix(f"_{test_name}")

            # 保存测试集错误样本
            self.logger.info(f"\n保存错误分类样本")
            self._save_test_misclassified_samples(test_df, y_true, y_pred, y_pred_proba, test_filepath)

        return results

    def _save_test_misclassified_samples(self, test_df: pd.DataFrame, y_true: np.ndarray,
                                         y_pred: np.ndarray, y_pred_proba: np.ndarray,
                                         test_filepath: str):
        """保存测试集中的错误分类样本"""
        misclassified_mask = (y_true != y_pred)
        if not any(misclassified_mask):
            self.logger.info("测试集所有样本预测正确，无错误分类样本")
            return

        self.logger.info(f"测试集发现 {sum(misclassified_mask)} 个错误分类样本")

        misclassified_samples = test_df[misclassified_mask].copy().reset_index(drop=True)

        # 添加预测信息
        y_true_mis = y_true[misclassified_mask]
        y_pred_mis = y_pred[misclassified_mask]
        y_pred_proba_mis = y_pred_proba[misclassified_mask]

        misclassified_samples['True_Label'] = y_true_mis
        misclassified_samples['Predicted_Label'] = y_pred_mis
        misclassified_samples['Prediction_Probability'] = y_pred_proba_mis
        misclassified_samples['Confidence'] = np.abs(y_pred_proba_mis - 0.5) * 2
        misclassified_samples['Error_Type'] = ['False_Positive' if true_label == 0 else 'False_Negative'
                                               for true_label in y_true_mis]
        misclassified_samples['Used_Threshold'] = self.optimal_threshold

        # 生成文件名
        test_name = os.path.splitext(os.path.basename(test_filepath))[0]
        filename = f'misclassified_samples_test_{test_name}.csv'
        save_path = os.path.join(self.config['misclassified_dir'], filename)

        success = self._save_csv(misclassified_samples, save_path)

        if success:
            false_positive_count = sum(misclassified_samples['Error_Type'] == 'False_Positive')
            false_negative_count = sum(misclassified_samples['Error_Type'] == 'False_Negative')

            self.logger.info(f"测试集错误分类样本数: {len(misclassified_samples)}")
            self.logger.info(f"测试集假阳性样本数: {false_positive_count}")
            self.logger.info(f"测试集假阴性样本数: {false_negative_count}")

        return misclassified_samples

    def save_model(self, filepath: Optional[str] = None) -> str:
        """保存模型和相关信息"""
        if self.model is None:
            self.logger.error("模型未训练")
            return ""

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.config['model_dir'], f'obstacle_classifier_{timestamp}.joblib')

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'optimal_threshold': self.optimal_threshold,
            'cv_scores': self.cv_scores,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"模型已保存到: {filepath}")

        # 保存模型信息摘要
        summary_file = filepath.replace('.joblib', '_summary.json')
        summary = {
            'timestamp': model_data['timestamp'],
            'optimal_threshold': self.optimal_threshold,
            'cv_scores': self.cv_scores,
            'feature_count': len(self.feature_names),
            'selected_feature_count': len(self.selected_features) if self.selected_features else len(
                self.feature_names),
            'config': self.config
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 保存特征列表
        feature_file = filepath.replace('.joblib', '_features.txt')
        with open(feature_file, 'w', encoding='utf-8') as f:
            f.write("模型特征列表:\n")
            features_to_use = self.selected_features if self.selected_features else self.feature_names
            for i, feature in enumerate(features_to_use, 1):
                f.write(f"{i}. {feature}\n")

        return filepath

    def load_model(self, filepath: str) -> bool:
        """加载模型"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data['feature_names']
            self.selected_features = model_data.get('selected_features', [])
            self.optimal_threshold = model_data.get('optimal_threshold', 0.5)
            self.cv_scores = model_data.get('cv_scores', {})

            if 'config' in model_data:
                self.config.update(model_data['config'])

            self.logger.info(f"模型加载成功: {filepath}")
            self.logger.info(f"训练时间: {model_data.get('timestamp', '未知')}")
            self.logger.info(f"特征数量: {len(self.feature_names)}")
            self.logger.info(f"最优阈值: {self.optimal_threshold}")
            if self.cv_scores:
                self.logger.info(f"交叉验证AUC: {self.cv_scores.get('auc', 'N/A')}")
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """生成模型报告"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.config['results_dir'], f'model_report_{timestamp}.md')

        report_content = f"""# 障碍物高度分类模型报告

## 模型基本信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 模型类型: LightGBM 二分类器
- 最优分类阈值: {self.optimal_threshold:.4f}

## 特征工程
- 总特征数: {len(self.feature_names)}
- 选择特征数: {len(self.selected_features) if self.selected_features else len(self.feature_names)}
- 特征选择: {'是' if self.config['feature_selection'] else '否'}

## 模型配置
```json
{json.dumps(self.config['model_params'], indent=2)}
```

## 交叉验证结果
"""
        if self.cv_scores:
            for metric, score in self.cv_scores.items():
                report_content += f"- {metric.upper()}: {score:.4f}\n"
        else:
            report_content += "- 未进行交叉验证\n"

        report_content += f"""
## 测试结果
"""
        if self.test_results:
            y_test = self.test_results['y_test']
            y_pred = self.test_results['y_pred']
            y_pred_proba = self.test_results['y_pred_proba']

            report_content += f"""- 测试样本数: {len(y_test)}
- 准确率: {accuracy_score(y_test, y_pred):.4f}
- AUC: {roc_auc_score(y_test, y_pred_proba):.4f}
- 错误样本数: {sum(y_test != y_pred)}
"""

        report_content += f"""
## 输出文件位置
- 模型文件: {self.config['model_dir']}
- 可视化图表: {self.config['plot_dir']}
- 预测结果: {self.config['results_dir']}
- 错误样本: {self.config['misclassified_dir']}
- 日志文件: {self.config['log_dir']}

## 使用说明
1. 加载模型: `classifier.load_model(model_path)`
2. 预测新数据: `classifier.predict(test_file_path)`
3. 查看错误样本: 检查 `{self.config['misclassified_dir']}` 目录
"""

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"模型报告已保存到: {output_path}")
        return output_path


def quick_run(train_file: str, test_file: Optional[str] = None, config: Optional[Dict] = None):
    """快速运行函数 - 改进版"""
    print("=== 障碍物高度分类器 - 改进版快速运行 ===\n")

    # 创建分类器
    classifier = ObstacleHeightClassifier(config)

    try:
        # 1. 训练阶段
        print("1. 数据加载和特征工程")
        train_df = classifier.load_data(train_file)
        train_df_processed = classifier.feature_engineering(train_df)

        print("\n2. 模型训练")
        classifier.train(train_df_processed)

        # 2. 特征重要性分析
        print("\n3. 特征重要性分析")
        classifier.plot_feature_importance()

        # 3. 预测阶段
        if test_file and os.path.exists(test_file):
            print("\n4. 预测阶段")
            results = classifier.predict(test_file)
        else:
            print("\n4. 跳过预测阶段（无测试文件）")
            results = None

        # 4. 生成报告
        print("\n5. 生成模型报告")
        report_path = classifier.generate_report()

        print(f"\n=== 运行完成 ===")
        print(f"详细日志请查看: {classifier.config['log_dir']}")
        print(f"完整报告请查看: {report_path}")

        return classifier

    except Exception as e:
        classifier.logger.error(f"运行失败: {e}")
        raise


if __name__ == "__main__":
    # ==================== 配置区域 ====================

    # 文件路径配置
    TRAIN_FILE = r'D:\PythonProject\data\processed_data\train_group1.csv'
    TEST_FILE = r'D:\PythonProject\data\processed_data\train_group2.csv'

    # 改进的配置
    OUTPUT_CONFIG = {
        # 路径配置
        'model_dir': r'D:\PythonProject\model\saved_model',
        'plot_dir': r'D:\PythonProject\results\visualization_results',
        'results_dir': r'D:\PythonProject\results\prediction_results',
        'misclassified_dir': r'D:\PythonProject\results\misclassified_results',
        'log_dir': r'D:\PythonProject\results\logs',
        'auto_create_dirs': True,
        'encoding': 'utf-8-sig',

        # 训练配置
        'test_size': 0.0,  # 使用全部数据训练
        'random_state': 42,
        'save_model': True,
        'cv_folds': 5,  # 5折交叉验证
        'threshold_optimization': True,  # 优化分类阈值
        'feature_selection': True,  # 特征选择
        'normalize_features': False,  # 是否标准化特征

        # 模型参数优化
        'model_params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,  # 增加复杂度
            'learning_rate': 0.03,  # 降低学习率
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,  # 防过拟合
            'lambda_l1': 0.1,  # L1正则化
            'lambda_l2': 0.1,  # L2正则化
            'verbose': -1,
            'is_unbalance': True,
            'random_state': 42
        }
    }

    # ==================== 运行区域 ====================

    # 运行改进版分类器
    classifier = quick_run(TRAIN_FILE, TEST_FILE, OUTPUT_CONFIG)

    print(f"\n📁 输出目录:")
    print(f"├── 模型保存: {OUTPUT_CONFIG['model_dir']}")
    print(f"├── 图片保存: {OUTPUT_CONFIG['plot_dir']}")
    print(f"├── 结果保存: {OUTPUT_CONFIG['results_dir']}")
    print(f"├── 错误样本: {OUTPUT_CONFIG['misclassified_dir']}")
    print(f"└── 日志文件: {OUTPUT_CONFIG['log_dir']}")

    # 检查关键文件
    print(f"\n🔍 关键文件检查:")

    # 检查错误样本文件
    misclassified_dir = OUTPUT_CONFIG['misclassified_dir']
    if os.path.exists(misclassified_dir):
        files = [f for f in os.listdir(misclassified_dir) if f.endswith('.csv')]
        if files:
            print(f"✅ 错误样本文件 ({len(files)} 个):")
            for file in files:
                file_path = os.path.join(misclassified_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"   └── {file} ({file_size} bytes)")
        else:
            print(f"⚠️  错误样本目录为空 - 可能模型预测100%正确")

    # 检查模型文件
    model_dir = OUTPUT_CONFIG['model_dir']
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        if model_files:
            latest_model = max(model_files)
            print(f"✅ 最新模型: {latest_model}")

    print(f"\n🎯 最优分类阈值: {classifier.optimal_threshold:.4f}")
    if classifier.cv_scores:
        print(f"📊 交叉验证AUC: {classifier.cv_scores.get('auc', 'N/A'):.4f}")