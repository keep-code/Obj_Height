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
    """障碍物高度分类器 - 改进版本"""

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
            'misclassified_dir': './misclassified',
            'auto_create_dirs': True,
            'encoding': 'utf-8-sig',
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
        for dir_path in [self.config['model_dir'], self.config['plot_dir'],
                         self.config['results_dir'], self.config['misclassified_dir']]:
            os.makedirs(dir_path, exist_ok=True)
            print(f"确保目录存在: {dir_path}")

    def _save_csv(self, df, filepath, encoding=None):
        """保存CSV文件，解决中文乱码问题"""
        encoding = encoding or self.config['encoding']
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False, encoding=encoding)
            print(f"文件已保存到: {filepath}")
            print(f"文件大小: {os.path.getsize(filepath)} bytes")
            return True
        except Exception as e:
            print(f"保存文件失败: {e}")
            try:
                df.to_csv(filepath, index=False, encoding='gbk')
                print(f"使用GBK编码保存到: {filepath}")
                return True
            except Exception as e2:
                try:
                    df.to_csv(filepath, index=False, encoding='utf-8')
                    print(f"使用UTF-8编码保存到: {filepath}")
                    return True
                except Exception as e3:
                    print(f"所有编码尝试失败: {e3}")
                    return False

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
        """特征工程 - 改进版本"""
        print("进行特征工程...")
        df_processed = df.copy()

        # ============ 关键改进：更严格的特征过滤 ============
        # 明确排除所有描述性和无关特征
        exclude_cols = [
            # 目标变量
            'HeightLabel',
            # 描述性信息（与障碍物高低无关）
            'Train_OD_Project',  # 项目名称
            'ObjName',  # 障碍物名称
            'Direction',  # 方向
            'Obj_ID',  # 障碍物ID - 新增排除
            'CurCyc',  # 当前周期 - 新增排除
            'TxSensID',  # 传感器ID - 新增排除
        ]

        print(f"排除的特征列: {exclude_cols}")

        # 验证排除的列是否存在于数据中
        existing_exclude_cols = [col for col in exclude_cols if col in df_processed.columns]
        print(f"实际排除的列: {existing_exclude_cols}")

        # 获取基础特征列（用于特征工程）
        base_features = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"基础特征列: {base_features}")

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

            # 新增特征
            'Avg_Echo_Strength': lambda x: (x['AvgDeEchoHigh_SameTx'] + x['AvgCeEchoHigh_SameTxRx']) / 2,
            'Distance_Ratio': lambda x: x['TrainObjDist'] / (x['AngleDist'] + 1e-8),
            'Echo_Strength_Ratio': lambda x: x['AvgDeEchoHigh_SameTx'] / (x['AvgCeEchoHigh_SameTxRx'] + 1e-8),
            'Odo_Stability': lambda x: x['OdoDiffObjDis'] / (x['OdoDiffDeDis'] + 1e-8),
        }

        # 应用特征工程
        for feature_name, operation in feature_ops.items():
            try:
                df_processed[feature_name] = operation(df_processed)
                print(f"创建特征: {feature_name}")
            except Exception as e:
                print(f"特征 {feature_name} 创建失败: {e}")

        # 更新特征列表 - 严格排除所有无关特征
        self.feature_names = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"特征工程完成，共 {len(self.feature_names)} 个特征")
        print(f"最终特征列表: {self.feature_names}")

        # ============ 新增：特征质量检查 ============
        self._check_feature_quality(df_processed)

        return df_processed

    def _check_feature_quality(self, df):
        """检查特征质量"""
        print("\n=== 特征质量检查 ===")

        # 检查缺失值
        missing_stats = df[self.feature_names].isnull().sum()
        if missing_stats.sum() > 0:
            print("发现缺失值:")
            print(missing_stats[missing_stats > 0])
        else:
            print("✓ 无缺失值")

        # 检查常数特征
        constant_features = []
        for feature in self.feature_names:
            if df[feature].nunique() <= 1:
                constant_features.append(feature)

        if constant_features:
            print(f"发现常数特征: {constant_features}")
            # 移除常数特征
            self.feature_names = [f for f in self.feature_names if f not in constant_features]
            print(f"移除常数特征后剩余: {len(self.feature_names)} 个特征")
        else:
            print("✓ 无常数特征")

        # 检查无穷值
        inf_features = []
        for feature in self.feature_names:
            if np.isinf(df[feature]).any():
                inf_features.append(feature)

        if inf_features:
            print(f"发现无穷值特征: {inf_features}")
            # 替换无穷值
            for feature in inf_features:
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                df[feature] = df[feature].fillna(df[feature].median())
            print("已替换无穷值为中位数")
        else:
            print("✓ 无无穷值")

    def train(self, train_df):
        """训练模型"""
        print("开始训练模型...")

        # 保存原始训练数据用于错误样本分析
        self.train_df_original = train_df.copy()

        # 准备数据
        X = train_df[self.feature_names]
        y = train_df['HeightLabel']

        print(f"训练数据形状: X{X.shape}, y{y.shape}")
        print(f"特征列表: {self.feature_names[:10]}...")  # 只显示前10个

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

        # LightGBM参数 - 优化后的参数
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
            'is_unbalance': True,
            'min_child_samples': 20,  # 防止过拟合
            'min_child_weight': 0.001,  # 防止过拟合
            'subsample_for_bin': 200000,  # 提高训练速度
            'reg_alpha': 0.1,  # L1正则化
            'reg_lambda': 0.1,  # L2正则化
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
        self._save_misclassified_samples(self.train_df_original)

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

    def _save_misclassified_samples(self, train_df=None, suffix=""):
        """保存错误分类样本"""
        if not self.test_results:
            print("无测试结果，跳过错误样本保存")
            return

        X_test = self.test_results['X_test']
        y_test = self.test_results['y_test']
        y_pred = self.test_results['y_pred']
        y_pred_proba = self.test_results['y_pred_proba']

        # 找出错误样本
        misclassified_mask = (y_test != y_pred)
        if not any(misclassified_mask):
            print("所有样本预测正确，无错误分类样本")
            return

        print(f"发现 {sum(misclassified_mask)} 个错误分类样本")

        # 获取错误样本的索引
        misclassified_indices = X_test[misclassified_mask].index

        # 如果有原始训练数据，从中提取完整信息
        if train_df is not None:
            misclassified_samples = train_df.loc[misclassified_indices].copy()
        else:
            misclassified_samples = X_test[misclassified_mask].copy()

        # 添加预测相关信息
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

        # 保存到指定目录
        filename = f'misclassified_samples{suffix}.csv'
        save_path = os.path.join(self.config['misclassified_dir'], filename)

        # 确保目录存在
        os.makedirs(self.config['misclassified_dir'], exist_ok=True)
        print(f"保存错误样本到: {save_path}")

        # 保存文件
        success = self._save_csv(misclassified_samples, save_path)

        if success:
            # 统计信息
            false_positive_count = sum(misclassified_samples['Error_Type'] == 'False_Positive')
            false_negative_count = sum(misclassified_samples['Error_Type'] == 'False_Negative')

            print(f"错误分类样本数: {len(misclassified_samples)}")
            print(f"假阳性样本数: {false_positive_count}")
            print(f"假阴性样本数: {false_negative_count}")
            print(f"平均预测置信度: {misclassified_samples['Confidence'].mean():.4f}")
        else:
            print("错误样本保存失败")

        return misclassified_samples

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

            # 保存测试集错误样本
            print(f"\n保存错误分类样本")
            self._save_test_misclassified_samples(test_df, y_true, y_pred, y_pred_proba, test_filepath)

        return results

    def _save_test_misclassified_samples(self, test_df, y_true, y_pred, y_pred_proba, test_filepath):
        """保存测试集中的错误分类样本"""
        misclassified_mask = (y_true != y_pred)
        if not any(misclassified_mask):
            print("测试集所有样本预测正确，无错误分类样本")
            return

        print(f"测试集发现 {sum(misclassified_mask)} 个错误分类样本")

        # 获取错误样本
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

        # 生成文件名
        test_name = os.path.splitext(os.path.basename(test_filepath))[0]
        filename = f'misclassified_samples_test_{test_name}.csv'
        save_path = os.path.join(self.config['misclassified_dir'], filename)

        # 确保目录存在
        os.makedirs(self.config['misclassified_dir'], exist_ok=True)
        print(f"保存测试集错误样本到: {save_path}")

        # 保存文件
        success = self._save_csv(misclassified_samples, save_path)

        if success:
            # 统计信息
            false_positive_count = sum(misclassified_samples['Error_Type'] == 'False_Positive')
            false_negative_count = sum(misclassified_samples['Error_Type'] == 'False_Negative')

            print(f"测试集错误分类样本数: {len(misclassified_samples)}")
            print(f"测试集假阳性样本数: {false_positive_count}")
            print(f"测试集假阴性样本数: {false_negative_count}")
        else:
            print("测试集错误样本保存失败")

        return misclassified_samples

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
            f.write("=" * 50 + "\n")
            f.write(f"总特征数: {len(self.feature_names)}\n")
            f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

            # 分类显示特征
            base_features = [f for f in self.feature_names if not any(
                keyword in f for keyword in ['_Ratio', '_Diff', '_Total', 'Avg_', 'Distance_', 'Odo_', 'Echo_'])]
            engineered_features = [f for f in self.feature_names if f not in base_features]

            f.write("基础特征:\n")
            for i, feature in enumerate(base_features, 1):
                f.write(f"{i:2d}. {feature}\n")

            f.write(f"\n工程特征 ({len(engineered_features)}个):\n")
            for i, feature in enumerate(engineered_features, 1):
                f.write(f"{i:2d}. {feature}\n")

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

    def compare_feature_importance_with_original(self):
        """与原始代码的特征重要性对比"""
        if self.model is None:
            print("模型未训练")
            return

        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # 检查是否包含了应该排除的特征
        problematic_features = ['Obj_ID', 'CurCyc', 'TxSensID']
        found_problematic = [f for f in problematic_features if f in self.feature_names]

        if found_problematic:
            print(f"⚠️  警告: 发现应该排除的特征: {found_problematic}")
        else:
            print("✅ 已正确排除所有描述性特征")

        print(f"\n特征重要性 Top 10:")
        print(feature_imp.head(10).to_string(index=False))

        return feature_imp


def quick_run(train_file, test_file=None, config=None):
    """快速运行函数 - 改进版本"""
    print("=== 障碍物高度分类器 - 改进版本 ===\n")

    # 创建分类器
    classifier = ObstacleHeightClassifier(config)

    # 训练
    print("1. 训练阶段")
    train_df = classifier.load_data(train_file)
    train_df_processed = classifier.feature_engineering(train_df)
    classifier.train(train_df_processed)

    # 特征重要性分析
    print("\n2. 特征重要性分析")
    classifier.plot_feature_importance()

    # 比较特征重要性
    print("\n3. 特征质量验证")
    classifier.compare_feature_importance_with_original()

    # 预测
    if test_file and os.path.exists(test_file):
        print("\n4. 预测阶段")
        results = classifier.predict(test_file)
    else:
        print("\n4. 跳过预测阶段（无测试文件）")

    print("\n=== 运行完成 ===")
    return classifier


def validate_features_exclusion(df):
    """验证特征排除是否正确"""
    print("=== 特征排除验证 ===")

    # 应该排除的特征
    should_exclude = [
        'HeightLabel', 'Train_OD_Project', 'ObjName', 'Direction',
        'Obj_ID', 'CurCyc', 'TxSensID'
    ]

    # 检查这些特征是否存在于数据中
    existing_features = df.columns.tolist()
    found_exclude = [col for col in should_exclude if col in existing_features]

    print(f"数据中存在的应排除特征: {found_exclude}")
    print(f"数据总列数: {len(existing_features)}")
    print(f"应排除的列数: {len(found_exclude)}")
    print(f"预期的特征列数: {len(existing_features) - len(found_exclude)}")

    # 显示前几行数据中这些特征的情况
    if found_exclude:
        print(f"\n这些特征的样例数据:")
        for col in found_exclude:
            if col in df.columns:
                unique_vals = df[col].unique()[:5]  # 只显示前5个唯一值
                print(f"  {col}: {unique_vals} (共{df[col].nunique()}个唯一值)")

    return found_exclude


if __name__ == "__main__":
    # ==================== 配置区域 ====================

    # 文件路径配置
    TRAIN_FILE = r'D:\PythonProject\data\processed_data\merged_train_data_fixed.csv'
    TEST_FILE = r'D:\PythonProject\data\processed_data\train_group2.csv'
    # TEST_FILE = r'D:\PythonProject\data\processed_data\merged_train_data_fixed.csv'

    # 输出路径配置
    OUTPUT_CONFIG = {
        'model_dir': r'D:\PythonProject\model\saved_model_improved',
        'plot_dir': r'D:\PythonProject\results\visualization_results_improved',
        'results_dir': r'D:\PythonProject\results\prediction_results_improved',
        'misclassified_dir': r'D:\PythonProject\results\misclassified_results_improved',
        'auto_create_dirs': True,
        'encoding': 'utf-8-sig',

        # 训练参数
        'test_size': 0.0,  # 使用全部数据训练
        'random_state': 42,
        'save_model': True  # 保存改进后的模型
    }

    print("🔍 首先验证数据中的特征...")

    # 读取数据并验证特征
    try:
        import pandas as pd

        df_sample = pd.read_csv(TEST_FILE, encoding='utf-8-sig')
        validate_features_exclusion(df_sample)
    except Exception as e:
        print(f"数据读取失败: {e}")

    print(f"\n{'=' * 60}")
    print("🚀 开始运行改进版分类器...")
    print(f"{'=' * 60}")

    # 运行改进版分类器
    classifier = quick_run(TRAIN_FILE, TEST_FILE, OUTPUT_CONFIG)

    print(f"\n📁 输出目录:")
    print(f"- 模型保存: {OUTPUT_CONFIG['model_dir']}")
    print(f"- 图片保存: {OUTPUT_CONFIG['plot_dir']}")
    print(f"- 结果保存: {OUTPUT_CONFIG['results_dir']}")
    print(f"- 错误样本: {OUTPUT_CONFIG['misclassified_dir']}")

    # print(f"\n✅ 改进完成！主要变化:")
    # print(f"   1. 排除了 Obj_ID, CurCyc, TxSensID 等描述性特征")
    # print(f"   2. 增加了特征质量检查")
    # print(f"   3. 优化了模型参数以防止过拟合")
    # print(f"   4. 增加了特征验证功能")