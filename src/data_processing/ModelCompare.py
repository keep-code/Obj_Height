import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class ModelComparisonTool:
    """模型对比验证工具"""

    def __init__(self):
        self.results = {}

    def prepare_features_original(self, df):
        """原始版本的特征准备"""
        exclude_cols = ['HeightLabel', 'Train_OD_Project', 'ObjName', 'Direction']

        # 创建工程特征（与原代码相同）
        df_processed = df.copy()

        feature_ops = {
            'DeEcho_Ratio': lambda x: x['PosDeDis1'] / (x['PosDeDis2'] + 1e-8),
            'CeEcho_Ratio': lambda x: x['PosCeDis1'] / (x['PosCeDis2'] + 1e-8),
            'DeAmp_Ratio': lambda x: x['PosDeAmp1'] / (x['PosDeAmp2'] + 1e-8),
            'CeAmp_Ratio': lambda x: x['PosCeAmp1'] / (x['PosCeAmp2'] + 1e-8),
            'Total_DeEcho': lambda x: x['PosDeDis1'] + x['PosDeDis2'],
            'Total_CeEcho': lambda x: x['PosCeDis1'] + x['PosCeDis2'],
            'Total_DeAmp': lambda x: x['PosDeAmp1'] + x['PosDeAmp2'],
            'Total_CeAmp': lambda x: x['PosCeAmp1'] + x['PosCeAmp2'],
            'DeDis_Diff': lambda x: x['PosDeDis1'] - x['PosDeDis2'],
            'CeDis_Diff': lambda x: x['PosCeDis1'] - x['PosCeDis2'],
            'DeAmp_Diff': lambda x: x['PosDeAmp1'] - x['PosDeAmp2'],
            'CeAmp_Diff': lambda x: x['PosCeAmp1'] - x['PosCeAmp2'],
            'Avg_Echo_Strength': lambda x: (x['AvgDeEchoHigh_SameTx'] + x['AvgCeEchoHigh_SameTxRx']) / 2,
            'Distance_Ratio': lambda x: x['TrainObjDist'] / (x['AngleDist'] + 1e-8)
        }

        for feature_name, operation in feature_ops.items():
            df_processed[feature_name] = operation(df_processed)

        feature_names = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"原始版本特征数: {len(feature_names)}")
        print(f"包含的问题特征: {[f for f in ['Obj_ID', 'CurCyc', 'TxSensID'] if f in feature_names]}")

        return df_processed[feature_names], feature_names

    def prepare_features_improved(self, df):
        """改进版本的特征准备"""
        exclude_cols = [
            'HeightLabel', 'Train_OD_Project', 'ObjName', 'Direction',
            'Obj_ID', 'CurCyc', 'TxSensID'  # 新增排除
        ]

        df_processed = df.copy()

        feature_ops = {
            'DeEcho_Ratio': lambda x: x['PosDeDis1'] / (x['PosDeDis2'] + 1e-8),
            'CeEcho_Ratio': lambda x: x['PosCeDis1'] / (x['PosCeDis2'] + 1e-8),
            'DeAmp_Ratio': lambda x: x['PosDeAmp1'] / (x['PosDeAmp2'] + 1e-8),
            'CeAmp_Ratio': lambda x: x['PosCeAmp1'] / (x['PosCeAmp2'] + 1e-8),
            'Total_DeEcho': lambda x: x['PosDeDis1'] + x['PosDeDis2'],
            'Total_CeEcho': lambda x: x['PosCeDis1'] + x['PosCeDis2'],
            'Total_DeAmp': lambda x: x['PosDeAmp1'] + x['PosDeAmp2'],
            'Total_CeAmp': lambda x: x['PosCeAmp1'] + x['PosCeAmp2'],
            'DeDis_Diff': lambda x: x['PosDeDis1'] - x['PosDeDis2'],
            'CeDis_Diff': lambda x: x['PosCeDis1'] - x['PosCeDis2'],
            'DeAmp_Diff': lambda x: x['PosDeAmp1'] - x['PosDeAmp2'],
            'CeAmp_Diff': lambda x: x['PosCeAmp1'] - x['PosCeAmp2'],
            'Avg_Echo_Strength': lambda x: (x['AvgDeEchoHigh_SameTx'] + x['AvgCeEchoHigh_SameTxRx']) / 2,
            'Distance_Ratio': lambda x: x['TrainObjDist'] / (x['AngleDist'] + 1e-8),
            'Echo_Strength_Ratio': lambda x: x['AvgDeEchoHigh_SameTx'] / (x['AvgCeEchoHigh_SameTxRx'] + 1e-8),
            'Odo_Stability': lambda x: x['OdoDiffObjDis'] / (x['OdoDiffDeDis'] + 1e-8),
        }

        for feature_name, operation in feature_ops.items():
            try:
                df_processed[feature_name] = operation(df_processed)
            except:
                pass

        feature_names = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"改进版本特征数: {len(feature_names)}")
        print(f"已排除问题特征: {[f for f in ['Obj_ID', 'CurCyc', 'TxSensID'] if f not in feature_names]}")

        return df_processed[feature_names], feature_names

    def cross_validation_comparison(self, df, cv_folds=5):
        """交叉验证对比两个版本"""
        print("=== 交叉验证对比 ===")

        # 准备数据
        X_original, features_original = self.prepare_features_original(df)
        X_improved, features_improved = self.prepare_features_improved(df)
        y = df['HeightLabel']

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
            'random_state': 42,
            'is_unbalance': True
        }

        # 交叉验证
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        results = {
            'original': {'accuracy': [], 'auc': []},
            'improved': {'accuracy': [], 'auc': []}
        }

        print(f"开始 {cv_folds} 折交叉验证...")

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_original, y)):
            print(f"\n第 {fold + 1} 折:")

            # 原始版本
            X_train_orig, X_val_orig = X_original.iloc[train_idx], X_original.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train_orig, label=y_train)
            # 修复：移除 verbose_eval 参数
            model_orig = lgb.train(params, train_data, num_boost_round=500)

            y_pred_orig = model_orig.predict(X_val_orig)
            y_pred_binary_orig = (y_pred_orig > 0.5).astype(int)

            acc_orig = accuracy_score(y_val, y_pred_binary_orig)
            auc_orig = roc_auc_score(y_val, y_pred_orig)

            results['original']['accuracy'].append(acc_orig)
            results['original']['auc'].append(auc_orig)

            # 改进版本
            X_train_imp, X_val_imp = X_improved.iloc[train_idx], X_improved.iloc[val_idx]

            train_data = lgb.Dataset(X_train_imp, label=y_train)
            # 修复：移除 verbose_eval 参数
            model_imp = lgb.train(params, train_data, num_boost_round=500)

            y_pred_imp = model_imp.predict(X_val_imp)
            y_pred_binary_imp = (y_pred_imp > 0.5).astype(int)

            acc_imp = accuracy_score(y_val, y_pred_binary_imp)
            auc_imp = roc_auc_score(y_val, y_pred_imp)

            results['improved']['accuracy'].append(acc_imp)
            results['improved']['auc'].append(auc_imp)

            print(f"  原始版本 - 准确率: {acc_orig:.4f}, AUC: {auc_orig:.4f}")
            print(f"  改进版本 - 准确率: {acc_imp:.4f}, AUC: {auc_imp:.4f}")

        # 计算统计结果
        self.results = results
        return results

    def analyze_results(self):
        """分析对比结果"""
        if not self.results:
            print("请先运行交叉验证")
            return

        results = self.results

        print("\n" + "=" * 60)
        print("=== 交叉验证结果分析 ===")
        print("=" * 60)

        # 计算统计值
        orig_acc_mean = np.mean(results['original']['accuracy'])
        orig_acc_std = np.std(results['original']['accuracy'])
        orig_auc_mean = np.mean(results['original']['auc'])
        orig_auc_std = np.std(results['original']['auc'])

        imp_acc_mean = np.mean(results['improved']['accuracy'])
        imp_acc_std = np.std(results['improved']['accuracy'])
        imp_auc_mean = np.mean(results['improved']['auc'])
        imp_auc_std = np.std(results['improved']['auc'])

        print("📊 准确率对比:")
        print(f"  原始版本: {orig_acc_mean:.4f} ± {orig_acc_std:.4f}")
        print(f"  改进版本: {imp_acc_mean:.4f} ± {imp_acc_std:.4f}")
        print(f"  差异: {imp_acc_mean - orig_acc_mean:+.4f}")

        print("\n📊 AUC对比:")
        print(f"  原始版本: {orig_auc_mean:.4f} ± {orig_auc_std:.4f}")
        print(f"  改进版本: {imp_auc_mean:.4f} ± {imp_auc_std:.4f}")
        print(f"  差异: {imp_auc_mean - orig_auc_mean:+.4f}")

        # 稳定性分析
        print(f"\n📈 模型稳定性:")
        print(f"  原始版本准确率标准差: {orig_acc_std:.4f}")
        print(f"  改进版本准确率标准差: {imp_acc_std:.4f}")
        stability_improvement = orig_acc_std - imp_acc_std
        print(f"  稳定性改进: {stability_improvement:+.4f} ({'更稳定' if stability_improvement > 0 else '稳定性下降'})")

        # 结论
        print(f"\n🎯 结论:")
        if imp_acc_mean > orig_acc_mean:
            print("  ✅ 改进版本性能更好")
        elif abs(imp_acc_mean - orig_acc_mean) < 0.01:
            print("  🟡 两版本性能相当，但改进版本避免了数据泄露风险")
        else:
            print("  🔍 原始版本性能看似更好，但需要进一步分析原因")

        print(f"\n💡 分析建议:")
        if orig_acc_mean > imp_acc_mean + 0.01:
            print("  1. 原始版本可能存在数据泄露，需要在独立测试集上验证")
            print("  2. 检查 Obj_ID、CurCyc、TxSensID 是否包含标签信息")
            print("  3. 改进版本虽然性能略低，但更可靠和可解释")
        else:
            print("  1. 改进版本成功消除了数据泄露风险")
            print("  2. 性能没有显著下降，模型更加可靠")

    def plot_comparison(self):
        """绘制对比图"""
        if not self.results:
            print("请先运行交叉验证")
            return

        results = self.results

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 准确率对比
        ax1.boxplot([results['original']['accuracy'], results['improved']['accuracy']],
                    labels=['original version', 'improved version'])
        ax1.set_title('AUC comparison')
        ax1.set_ylabel('AUC')
        ax1.grid(True, alpha=0.3)

        # AUC对比
        ax2.boxplot([results['original']['auc'], results['improved']['auc']],
                    labels=['original version', 'improved version'])
        ax2.set_title('AUC comparison')
        ax2.set_ylabel('AUC')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def feature_importance_analysis(self, df):
        """特征重要性分析"""
        print("\n=== 特征重要性分析 ===")

        X_original, features_original = self.prepare_features_original(df)
        X_improved, features_improved = self.prepare_features_improved(df)
        y = df['HeightLabel']

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True
        }

        # 训练两个模型
        train_data_orig = lgb.Dataset(X_original, label=y)
        # 修复：移除 verbose_eval 参数
        model_orig = lgb.train(params, train_data_orig, num_boost_round=500)

        train_data_imp = lgb.Dataset(X_improved, label=y)
        # 修复：移除 verbose_eval 参数
        model_imp = lgb.train(params, train_data_imp, num_boost_round=500)

        # 获取特征重要性
        importance_orig = model_orig.feature_importance(importance_type='gain')
        importance_imp = model_imp.feature_importance(importance_type='gain')

        # 分析原始版本中问题特征的重要性
        problematic_features = ['Obj_ID', 'CurCyc', 'TxSensID']

        print("原始版本中问题特征的重要性:")
        for i, feature in enumerate(features_original):
            if feature in problematic_features:
                importance_rank = sorted(enumerate(importance_orig), key=lambda x: x[1], reverse=True)
                rank = next((idx for idx, (feat_idx, _) in enumerate(importance_rank) if feat_idx == i), -1)
                print(f"  {feature}: 重要性={importance_orig[i]:.1f}, 排名={rank + 1}/{len(features_original)}")

        # 显示改进版本 Top 特征
        feature_imp_df = pd.DataFrame({
            'feature': features_improved,
            'importance': importance_imp
        }).sort_values('importance', ascending=False)

        print(f"\n改进版本 Top 10 重要特征:")
        print(feature_imp_df.head(10).to_string(index=False))

        return model_orig, model_imp


def run_comparison(data_file):
    """运行完整对比分析"""
    print("🔬 模型对比分析工具")
    print("=" * 50)

    # 加载数据
    try:
        df = pd.read_csv(data_file, encoding='utf-8-sig')
        print(f"✅ 数据加载成功: {df.shape}")
    except:
        try:
            df = pd.read_csv(data_file, encoding='gbk')
            print(f"✅ 数据加载成功: {df.shape}")
        except:
            df = pd.read_csv(data_file)
            print(f"✅ 数据加载成功: {df.shape}")

    # 创建对比工具
    comparator = ModelComparisonTool()

    # 运行对比分析
    print(f"\n1️⃣ 交叉验证对比")
    results = comparator.cross_validation_comparison(df, cv_folds=5)

    print(f"\n2️⃣ 结果分析")
    comparator.analyze_results()

    print(f"\n3️⃣ 绘制对比图")
    comparator.plot_comparison()

    print(f"\n4️⃣ 特征重要性分析")
    model_orig, model_imp = comparator.feature_importance_analysis(df)

    return comparator, model_orig, model_imp


if __name__ == "__main__":
    # 使用示例
    DATA_FILE = r'D:\PythonProject\data\processed_data\merged_train_data_fixed.csv'

    comparator, model_orig, model_imp = run_comparison(DATA_FILE)