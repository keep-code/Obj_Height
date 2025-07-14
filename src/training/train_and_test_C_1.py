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
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LightGBMToCConverter:
    """LightGBM模型转C语言代码生成器"""

    def __init__(self, model, feature_names, output_dir="./c_model"):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = output_dir
        self.num_features = len(feature_names)
        os.makedirs(output_dir, exist_ok=True)

    def extract_model_info(self):
        """提取模型信息"""
        model_dict = self.model.dump_model()
        self.num_trees = model_dict['num_tree_per_iteration']
        self.trees = model_dict['tree_info']
        self.objective = model_dict.get('objective', 'binary')

        print(f"模型信息:")
        print(f"- 树的数量: {len(self.trees)}")
        print(f"- 特征数量: {self.num_features}")
        print(f"- 目标函数: {self.objective}")

        return model_dict

    def generate_tree_function(self, tree_dict, tree_id):
        """生成单个树的C函数"""

        def generate_node_code(node, indent=0):
            spaces = "    " * indent

            if 'leaf_value' in node:
                return f"{spaces}return {node['leaf_value']:.6f};\n"
            else:
                feature_idx = node['split_feature']
                threshold = node['threshold']
                left_child = node['left_child']
                right_child = node['right_child']

                code = f"{spaces}if (features[{feature_idx}] <= {threshold:.6f}) {{\n"
                code += generate_node_code(left_child, indent + 1)
                code += f"{spaces}}} else {{\n"
                code += generate_node_code(right_child, indent + 1)
                code += f"{spaces}}}\n"

                return code

        func_code = f"""
static double tree_{tree_id}(const double *features) {{
{generate_node_code(tree_dict['tree_structure'], 1)}}}
"""
        return func_code

    def generate_prediction_function(self):
        """生成预测函数"""
        tree_calls = []
        for i in range(len(self.trees)):
            tree_calls.append(f"    sum += tree_{i}(features);")

        tree_calls_str = "\n".join(tree_calls)

        prediction_code = f"""
double predict_raw(const double *features) {{
    double sum = 0.0;
{tree_calls_str}
    return sum;
}}

double predict_probability(const double *features) {{
    double raw_score = predict_raw(features);
    return 1.0 / (1.0 + exp(-raw_score));  // sigmoid函数
}}

int predict_class(const double *features) {{
    return predict_probability(features) > 0.5 ? 1 : 0;
}}

double predict_confidence(const double *features) {{
    double prob = predict_probability(features);
    return prob > 0.5 ? prob : (1.0 - prob);  // 置信度
}}
"""
        return prediction_code

    def generate_header_file(self):
        """生成头文件"""
        header_content = f"""#ifndef OBSTACLE_HEIGHT_MODEL_H
#define OBSTACLE_HEIGHT_MODEL_H

#include <math.h>

// 模型信息
#define NUM_FEATURES {self.num_features}
#define NUM_TREES {len(self.trees)}

// 特征索引定义
"""
        for i, feature_name in enumerate(self.feature_names):
            feature_macro = feature_name.upper().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            header_content += f"#define FEATURE_{feature_macro} {i}\n"

        header_content += f"""
// 函数声明
double predict_raw(const double *features);
double predict_probability(const double *features);
int predict_class(const double *features);
double predict_confidence(const double *features);

// 预测结果结构体
typedef struct {{
    int height_label;          // 高度标签 (0=低障碍物, 1=高障碍物)
    double probability;        // 预测概率
    double confidence;         // 置信度
    double raw_score;         // 原始分数
}} ObstacleHeightPrediction;

ObstacleHeightPrediction predict_obstacle_height(const double *features);

#endif // OBSTACLE_HEIGHT_MODEL_H
"""
        return header_content

    def generate_source_file(self):
        """生成源文件"""
        model_info = self.extract_model_info()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_content = f"""/*
 * 障碍物高度分类器 - LightGBM模型C语言实现
 * 生成时间: {timestamp}
 * 特征数量: {self.num_features}
 * 树的数量: {len(self.trees)}
 * 
 * 使用说明:
 * - 输入: double数组，长度为NUM_FEATURES，包含所有特征值
 * - 输出: ObstacleHeightPrediction结构体，包含预测结果和置信度
 */

#include "obstacle_height_model.h"
#include <math.h>

"""

        # 生成所有树的函数
        for i, tree in enumerate(self.trees):
            source_content += self.generate_tree_function(tree, i)

        # 生成预测函数
        source_content += self.generate_prediction_function()

        # 生成便利函数
        source_content += """
ObstacleHeightPrediction predict_obstacle_height(const double *features) {
    ObstacleHeightPrediction result;
    result.raw_score = predict_raw(features);
    result.probability = predict_probability(features);
    result.height_label = predict_class(features);
    result.confidence = predict_confidence(features);
    return result;
}
"""

        return source_content

    def generate_test_file(self):
        """生成测试文件"""
        test_content = f"""/*
 * 障碍物高度分类器测试程序
 */

#include <stdio.h>
#include <stdlib.h>
#include "obstacle_height_model.h"

int main() {{
    printf("=== 障碍物高度分类器 C语言版本 ===\\n");
    printf("特征数量: %d\\n", NUM_FEATURES);
    printf("树的数量: %d\\n", NUM_TREES);
    printf("\\n");

    // 示例特征数据 (请根据实际情况修改)
    double features[NUM_FEATURES] = {{0}};

    // 示例1: 设置一些示例特征值
    printf("示例1: 测试数据\\n");
"""

        # 为前几个特征设置示例值
        for i in range(min(5, self.num_features)):
            test_content += f"    features[{i}] = {100.0 * (i + 1):.1f};  // {self.feature_names[i]}\n"

        test_content += f"""    
    // 进行预测
    ObstacleHeightPrediction result = predict_obstacle_height(features);

    printf("预测结果:\\n");
    printf("- 高度标签: %d (%s)\\n", result.height_label, 
           result.height_label ? "高障碍物" : "低障碍物");
    printf("- 概率: %.6f\\n", result.probability);
    printf("- 置信度: %.6f\\n", result.confidence);
    printf("- 原始分数: %.6f\\n", result.raw_score);
    printf("\\n");

    // 示例2: 另一组测试数据
    printf("示例2: 另一组测试数据\\n");
    for(int i = 0; i < NUM_FEATURES; i++) {{
        features[i] = (double)(i * 50 + 200);  // 简单的测试数据
    }}

    result = predict_obstacle_height(features);
    printf("预测结果:\\n");
    printf("- 高度标签: %d (%s)\\n", result.height_label, 
           result.height_label ? "高障碍物" : "低障碍物");
    printf("- 概率: %.6f\\n", result.probability);
    printf("- 置信度: %.6f\\n", result.confidence);
    printf("- 原始分数: %.6f\\n", result.raw_score);

    return 0;
}}
"""
        return test_content

    def convert(self):
        """执行转换"""
        print(f"🚀 开始转换LightGBM模型到C语言...")
        print(f"📁 输出目录: {self.output_dir}")

        # 生成头文件
        header_content = self.generate_header_file()
        header_file = os.path.join(self.output_dir, "obstacle_height_model.h")
        with open(header_file, 'w') as f:
            f.write(header_content)
        print(f"✅ 生成头文件: {header_file}")

        # 生成源文件
        source_content = self.generate_source_file()
        source_file = os.path.join(self.output_dir, "obstacle_height_model.c")
        with open(source_file, 'w') as f:
            f.write(source_content)
        print(f"✅ 生成源文件: {source_file}")

        # 生成测试文件
        test_content = self.generate_test_file()
        test_file = os.path.join(self.output_dir, "test_obstacle_height.c")
        with open(test_file, 'w') as f:
            f.write(test_content)
        print(f"✅ 生成测试文件: {test_file}")

        # 生成Makefile
        makefile_content = self._generate_makefile()
        makefile_file = os.path.join(self.output_dir, "Makefile")
        with open(makefile_file, 'w') as f:
            f.write(makefile_content)
        print(f"✅ 生成Makefile: {makefile_file}")

        # 保存特征映射
        self._save_feature_mapping()
        print(f"✅ 保存特征映射")

        # 生成使用说明
        self._generate_usage_guide()

        print(f"\n🎉 转换完成!")
        print(f"📂 所有文件已保存到: {self.output_dir}")
        print(f"🔨 编译命令: cd {self.output_dir} && make")
        print(f"🚀 运行测试: cd {self.output_dir} && ./test_obstacle_height")

    def _generate_makefile(self):
        """生成Makefile"""
        return """# 障碍物高度分类器 C语言版本 Makefile

CC = gcc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

TARGET = test_obstacle_height
SOURCES = obstacle_height_model.c test_obstacle_height.c
HEADERS = obstacle_height_model.h

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
"""

    def _save_feature_mapping(self):
        """保存特征映射信息"""
        feature_mapping = {
            "feature_names": self.feature_names,
            "feature_count": self.num_features,
            "feature_indices": {name: i for i, name in enumerate(self.feature_names)},
            "model_info": {
                "trees": len(self.trees),
                "objective": self.objective
            }
        }

        mapping_file = os.path.join(self.output_dir, "feature_mapping.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(feature_mapping, f, indent=2, ensure_ascii=False)

        # 生成特征列表文件
        readme_file = os.path.join(self.output_dir, "feature_list.txt")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("障碍物高度分类器特征列表\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"特征总数: {self.num_features}\n")
            f.write(f"树的数量: {len(self.trees)}\n\n")
            f.write("特征索引和名称:\n")
            for i, name in enumerate(self.feature_names):
                f.write(f"{i:2d}: {name}\n")

    def _generate_usage_guide(self):
        """生成使用说明"""
        guide_content = f"""# 障碍物高度分类器 C语言版本

## 项目说明
本项目将训练好的LightGBM障碍物高度分类模型转换为C语言实现，可以部署到嵌入式系统或其他C/C++项目中。

## 文件结构
- `obstacle_height_model.h` - 头文件，包含函数声明和宏定义
- `obstacle_height_model.c` - 源文件，包含模型实现
- `test_obstacle_height.c` - 测试程序
- `Makefile` - 编译配置文件
- `feature_mapping.json` - 特征映射信息
- `feature_list.txt` - 特征列表
- `README.md` - 使用说明

## 快速开始

### 编译和测试
```bash
# 编译
make

# 运行测试
./test_obstacle_height

# 清理编译文件
make clean
```

### 在您的项目中集成

1. 将以下文件复制到您的项目中：
   - `obstacle_height_model.h`
   - `obstacle_height_model.c`

2. 在您的代码中包含头文件：
   ```c
   #include "obstacle_height_model.h"
   ```

3. 编译时链接数学库：
   ```bash
   gcc -o your_program your_program.c obstacle_height_model.c -lm
   ```

## API 使用说明

### 核心函数
- `predict_obstacle_height(features)` - 主要预测函数，返回完整结果
- `predict_class(features)` - 仅返回分类结果 (0或1)
- `predict_probability(features)` - 返回预测概率 (0.0-1.0)
- `predict_confidence(features)` - 返回置信度 (0.5-1.0)

### 使用示例
```c
#include "obstacle_height_model.h"
#include <stdio.h>

int main() {{
    // 准备特征数据 (共{self.num_features}个特征)
    double features[NUM_FEATURES];

    // 设置特征值 (根据您的实际传感器数据)"""

        # 添加特征示例
        for i, feature_name in enumerate(self.feature_names[:5]):  # 只显示前5个作为示例
            guide_content += f"""
    features[{i}] = your_sensor_data_{i};  // {feature_name}"""

        if len(self.feature_names) > 5:
            guide_content += f"""
    // ... 设置剩余的 {len(self.feature_names) - 5} 个特征"""

        guide_content += f"""

    // 进行预测
    ObstacleHeightPrediction result = predict_obstacle_height(features);

    // 使用预测结果
    if (result.height_label == 1) {{
        printf("检测到高障碍物！概率: %.2f, 置信度: %.2f\\n", 
               result.probability, result.confidence);
        // 执行高障碍物处理逻辑
    }} else {{
        printf("检测到低障碍物。概率: %.2f, 置信度: %.2f\\n", 
               1.0 - result.probability, result.confidence);
        // 执行低障碍物处理逻辑
    }}

    return 0;
}}
```

## 特征说明

模型需要 {self.num_features} 个特征输入，特征索引和含义如下：

| 索引 | 特征名称 | 说明 |
|------|----------|------|"""

        for i, name in enumerate(self.feature_names):
            guide_content += f"""
| {i} | {name} | 传感器特征 |"""

        guide_content += f"""

## 预测结果说明

### ObstacleHeightPrediction 结构体
```c
typedef struct {{
    int height_label;      // 0=低障碍物, 1=高障碍物
    double probability;    // 预测为高障碍物的概率 (0.0-1.0)
    double confidence;     // 预测置信度 (0.5-1.0)
    double raw_score;      // 模型原始输出分数
}} ObstacleHeightPrediction;
```

### 结果解释
- **height_label**: 最终分类结果
  - 0: 低障碍物
  - 1: 高障碍物
- **probability**: 预测为高障碍物的概率
  - 越接近1.0，越可能是高障碍物
  - 越接近0.0，越可能是低障碍物
- **confidence**: 预测置信度
  - 范围 0.5-1.0
  - 越接近1.0，预测越可靠
- **raw_score**: 模型原始输出，可用于调试

## 性能优化建议

1. **特征预处理**: 确保输入特征的数值范围与训练时一致
2. **内存使用**: 模型为静态实现，不需要动态内存分配
3. **计算复杂度**: O(树的数量 × 平均树深度)
4. **线程安全**: 所有函数都是线程安全的

## 注意事项

1. 特征数组必须包含 {self.num_features} 个double类型的值
2. 特征顺序必须与训练时保持一致
3. 需要链接数学库 (-lm)
4. 输入特征值应在合理范围内，避免极端值

## 故障排除

### 编译错误
- 确保包含了 `<math.h>` 头文件
- 确保链接了数学库 (`-lm`)

### 预测结果异常
- 检查特征值是否在合理范围内
- 确认特征顺序与训练时一致
- 验证输入数据格式正确

## 模型信息

- **特征数量**: {self.num_features}
- **树的数量**: {len(self.trees) if hasattr(self, 'trees') else 'N/A'}
- **目标函数**: 二分类 (binary classification)
- **输出类别**: 0 (低障碍物), 1 (高障碍物)
"""

        guide_file = os.path.join(self.output_dir, "README.md")
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        print(f"✅ 生成使用指南: {guide_file}")


class ObstacleHeightClassifier:
    """障碍物高度分类器 - 增强版本（含C语言转换功能）"""

    def __init__(self, config=None):
        # 默认配置
        self.config = {
            'model_dir': './models',
            'plot_dir': './plots',
            'results_dir': './results',
            'misclassified_dir': './misclassified',
            'c_model_dir': './c_model',  # 新增：C语言模型输出目录
            'auto_create_dirs': True,
            'encoding': 'utf-8-sig',
            'test_size': 0.3,
            'random_state': 42,
            'save_model': True,
            'auto_convert_to_c': True,  # 新增：自动转换为C语言
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
        dirs = [
            self.config['model_dir'],
            self.config['plot_dir'],
            self.config['results_dir'],
            self.config['misclassified_dir'],
            self.config['c_model_dir']  # 新增C语言模型目录
        ]
        for dir_path in dirs:
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

        # 排除描述性和无关特征
        exclude_cols = [
            'HeightLabel',
            'Train_OD_Project',
            'ObjName',
            'Direction',
            'Obj_ID',
            'CurCyc',
            'TxSensID',
        ]

        print(f"排除的特征列: {exclude_cols}")

        # 获取基础特征列
        base_features = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"基础特征列: {base_features}")

        # 创建新特征
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

        # 应用特征工程
        for feature_name, operation in feature_ops.items():
            try:
                df_processed[feature_name] = operation(df_processed)
                print(f"创建特征: {feature_name}")
            except Exception as e:
                print(f"特征 {feature_name} 创建失败: {e}")

        # 更新特征列表
        self.feature_names = [col for col in df_processed.columns if col not in exclude_cols]
        print(f"特征工程完成，共 {len(self.feature_names)} 个特征")

        # 特征质量检查
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
            for feature in inf_features:
                df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
                df[feature] = df[feature].fillna(df[feature].median())
            print("已替换无穷值为中位数")
        else:
            print("✓ 无无穷值")

    def train(self, train_df):
        """训练模型"""
        print("开始训练模型...")

        # 保存原始训练数据
        self.train_df_original = train_df.copy()

        # 准备数据
        X = train_df[self.feature_names]
        y = train_df['HeightLabel']

        print(f"训练数据形状: X{X.shape}, y{y.shape}")

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
            'is_unbalance': True,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
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
            model_path = self.save_model()

            # 自动转换为C语言
            if self.config['auto_convert_to_c']:
                print(f"\n🔄 开始自动转换为C语言...")
                self.convert_to_c_language()

        return self.test_results if X_test is not None else None

    def convert_to_c_language(self):
        """将训练好的模型转换为C语言代码"""
        if self.model is None:
            print("❌ 模型未训练，无法转换")
            return False

        try:
            print(f"📝 转换LightGBM模型为C语言代码...")

            # 创建转换器
            converter = LightGBMToCConverter(
                model=self.model,
                feature_names=self.feature_names,
                output_dir=self.config['c_model_dir']
            )

            # 执行转换
            converter.convert()

            print(f"\n🎉 C语言转换完成!")
            print(f"📁 C语言代码保存在: {self.config['c_model_dir']}")
            print(f"🔨 编译命令: cd {self.config['c_model_dir']} && make")
            print(f"🚀 运行测试: cd {self.config['c_model_dir']} && ./test_obstacle_height")

            return True

        except Exception as e:
            print(f"❌ C语言转换失败: {e}")
            return False

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

        return filepath

    def _save_misclassified_samples(self, train_df=None, suffix=""):
        """保存错误分类样本"""
        if not self.test_results:
            print("无测试结果，跳过错误样本保存")
            return

        X_test = self.test_results['X_test']
        y_test = self.test_results['y_test']
        y_pred = self.test_results['y_pred']
        y_pred_proba = self.test_results['y_pred_proba']

        misclassified_mask = (y_test != y_pred)
        if not any(misclassified_mask):
            print("所有样本预测正确，无错误分类样本")
            return

        print(f"发现 {sum(misclassified_mask)} 个错误分类样本")

        misclassified_indices = X_test[misclassified_mask].index

        if train_df is not None:
            misclassified_samples = train_df.loc[misclassified_indices].copy()
        else:
            misclassified_samples = X_test[misclassified_mask].copy()

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

        filename = f'misclassified_samples{suffix}.csv'
        save_path = os.path.join(self.config['misclassified_dir'], filename)

        os.makedirs(self.config['misclassified_dir'], exist_ok=True)
        success = self._save_csv(misclassified_samples, save_path)

        if success:
            false_positive_count = sum(misclassified_samples['Error_Type'] == 'False_Positive')
            false_negative_count = sum(misclassified_samples['Error_Type'] == 'False_Negative')

            print(f"错误分类样本数: {len(misclassified_samples)}")
            print(f"假阳性样本数: {false_positive_count}")
            print(f"假阴性样本数: {false_negative_count}")

        return misclassified_samples


# 快速运行函数 - 增强版本
def quick_run_with_c_conversion(train_file, test_file=None, config=None):
    """快速运行函数 - 含自动C语言转换"""
    print("=== 障碍物高度分类器 - 增强版本（含C语言转换）===\n")

    # 默认配置
    default_config = {
        'model_dir': './models_enhanced',
        'plot_dir': './plots_enhanced',
        'results_dir': './results_enhanced',
        'misclassified_dir': './misclassified_enhanced',
        'c_model_dir': './c_model_enhanced',
        'auto_convert_to_c': True,
        'test_size': 0.0,  # 使用全部数据训练
        'random_state': 42,
    }

    if config:
        default_config.update(config)

    # 创建分类器
    classifier = ObstacleHeightClassifier(default_config)

    # 训练
    print("1. 数据加载和特征工程")
    train_df = classifier.load_data(train_file)
    train_df_processed = classifier.feature_engineering(train_df)

    print("\n2. 模型训练")
    classifier.train(train_df_processed)

    # 特征重要性分析
    print("\n3. 特征重要性分析")
    feature_imp = classifier.plot_feature_importance()

    # 预测测试数据
    if test_file and os.path.exists(test_file):
        print("\n4. 测试数据预测")
        results = classifier.predict(test_file)
    else:
        print("\n4. 跳过预测阶段（无测试文件）")

    print("\n=== 运行完成 ===")
    print(f"📊 Python模型文件: {classifier.config['model_dir']}")
    print(f"🔧 C语言代码: {classifier.config['c_model_dir']}")

    return classifier


def predict_with_c_model(c_model_dir, features):
    """
    使用生成的C语言模型进行预测的Python包装函数

    Parameters:
    -----------
    c_model_dir : str
        C语言模型目录
    features : list or numpy.array
        特征数组

    Returns:
    --------
    dict : 预测结果
    """
    import subprocess
    import tempfile
    import json

    try:
        # 创建临时的C程序来调用模型
        temp_c_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include "obstacle_height_model.h"

int main() {{
    double features[NUM_FEATURES] = {{{', '.join(map(str, features))}}};

    ObstacleHeightPrediction result = predict_obstacle_height(features);

    // 输出JSON格式结果
    printf("{{");
    printf("\\"height_label\\": %d,", result.height_label);
    printf("\\"probability\\": %.6f,", result.probability);
    printf("\\"confidence\\": %.6f,", result.confidence);
    printf("\\"raw_score\\": %.6f", result.raw_score);
    printf("}}");

    return 0;
}}
"""

        # 保存临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(temp_c_code)
            temp_c_file = f.name

        # 编译和运行
        temp_exe = temp_c_file.replace('.c', '')

        compile_cmd = [
            'gcc', '-o', temp_exe, temp_c_file,
            os.path.join(c_model_dir, 'obstacle_height_model.c'),
            '-I', c_model_dir, '-lm'
        ]

        subprocess.run(compile_cmd, check=True, capture_output=True)
        result = subprocess.run([temp_exe], capture_output=True, text=True, check=True)

        # 解析结果
        prediction = json.loads(result.stdout)

        # 清理临时文件
        os.unlink(temp_c_file)
        os.unlink(temp_exe)

        return prediction

    except Exception as e:
        print(f"C模型预测失败: {e}")
        return None


def convert_existing_model_to_c(model_path, output_dir="./c_model_converted"):
    """
    转换已保存的.joblib模型文件为C语言代码

    Parameters:
    -----------
    model_path : str
        .joblib模型文件路径
    output_dir : str
        输出目录
    """
    try:
        # 加载模型
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            model = model_data['model']
            feature_names = model_data.get('feature_names', [])
        else:
            model = model_data
            feature_names = []

        print(f"📂 加载模型: {model_path}")
        print(f"🎯 输出目录: {output_dir}")

        # 创建转换器并执行转换
        converter = LightGBMToCConverter(model, feature_names, output_dir)
        converter.convert()

        return True

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False


if __name__ == "__main__":
    # ==================== 配置区域 ====================

    # 文件路径配置
    TRAIN_FILE = r'D:\PythonProject\data\processed_data\train_group1.csv'
    TEST_FILE = r'D:\PythonProject\data\processed_data\train_group2.csv'

    # 输出路径配置
    OUTPUT_CONFIG = {
        'model_dir': r'D:\PythonProject\model\saved_model_enhanced',
        'plot_dir': r'D:\PythonProject\results\visualization_results_enhanced',
        'results_dir': r'D:\PythonProject\results\prediction_results_enhanced',
        'misclassified_dir': r'D:\PythonProject\results\misclassified_results_enhanced',
        'c_model_dir': r'D:\PythonProject\c_model',  # C语言模型输出目录
        'auto_create_dirs': True,
        'encoding': 'utf-8-sig',
        'test_size': 0.0,  # 使用全部数据训练
        'random_state': 42,
        'save_model': True,
        'auto_convert_to_c': True,  # 启用自动转换为C语言
    }

    print(f"🚀 开始运行增强版障碍物分类器...")
    print(f"📁 C语言模型将保存在: {OUTPUT_CONFIG['c_model_dir']}")
    print(f"{'=' * 80}")

    # 运行增强版分类器（自动转换为C语言）
    classifier = quick_run_with_c_conversion(TRAIN_FILE, TEST_FILE, OUTPUT_CONFIG)

    print(f"\n{'=' * 80}")
    print(f"✅ 训练和转换完成！")
    print(f"\n📊 Python模型和结果:")
    print(f"   - 模型文件: {OUTPUT_CONFIG['model_dir']}")
    print(f"   - 可视化结果: {OUTPUT_CONFIG['plot_dir']}")
    print(f"   - 预测结果: {OUTPUT_CONFIG['results_dir']}")
    print(f"   - 错误样本: {OUTPUT_CONFIG['misclassified_dir']}")

    print(f"\n🔧 C语言模型:")
    print(f"   - C代码目录: {OUTPUT_CONFIG['c_model_dir']}")
    print(f"   - 编译命令: cd {OUTPUT_CONFIG['c_model_dir']} && make")
    print(f"   - 运行测试: cd {OUTPUT_CONFIG['c_model_dir']} && ./test_obstacle_height")

    print(f"\n🎯 主要改进:")
    print(f"   ✓ 自动生成C语言代码")
    print(f"   ✓ 包含完整的编译和测试文件")
    print(f"   ✓ 详细的使用文档和示例")
    print(f"   ✓ 优化的特征工程和模型参数")

    # 示例：如何单独转换已有模型
    print(f"\n💡 提示：如果你有已保存的.joblib模型文件，可以这样转换:")
    print(f"   convert_existing_model_to_c('path/to/your/model.joblib', './output_dir')")

    # 示例：如何使用生成的C模型进行预测
    print(f"\n🔮 C模型预测示例:")
    print(f"   features = [100.0, 200.0, ...]  # 你的特征数据")
    print(f"   result = predict_with_c_model('{OUTPUT_CONFIG['c_model_dir']}', features)")
    print(f"   print(result)  # {{'height_label': 1, 'probability': 0.85, ...}}")


# 使用说明和测试函数
def test_c_model_integration():
    """测试C语言模型集成"""
    print("🧪 测试C语言模型集成...")

    # 示例特征数据（需要根据实际情况调整）
    sample_features = [
        100.0, 200.0, 150.0, 250.0,  # PosDeDis1, PosDeAmp1, PosDeDis2, PosDeAmp2
        120.0, 180.0, 110.0, 190.0,  # PosCeDis1, PosCeAmp1, PosCeDis2, PosCeAmp2
        500.0, 300.0, 400.0,  # TrainObjDist, AvgDeEchoHigh_SameTx, AvgCeEchoHigh_SameTxRx
        50.0, 30.0, 20.0, 40.0,  # OdoDiffObjDis, OdoDiffDeDis, OdoDiff, ObjDiff
        800.0, 25.0, 450.0  # CosAngle, RateOfVhoDeDiff, AngleDist
        # 加上工程特征会自动计算
    ]

    # 如果C模型目录存在，尝试预测
    c_model_dir = "./c_model"
    if os.path.exists(os.path.join(c_model_dir, "obstacle_height_model.h")):
        try:
            result = predict_with_c_model(c_model_dir, sample_features)
            if result:
                print(f"✅ C模型预测成功:")
                print(
                    f"   - 高度标签: {result['height_label']} ({'高障碍物' if result['height_label'] else '低障碍物'})")
                print(f"   - 概率: {result['probability']:.4f}")
                print(f"   - 置信度: {result['confidence']:.4f}")
            else:
                print("❌ C模型预测失败")
        except Exception as e:
            print(f"⚠️  C模型测试跳过: {e}")
    else:
        print(f"⚠️  C模型目录不存在: {c_model_dir}")
        print(f"   请先运行主程序生成C模型")


# 如果直接运行此文件，执行测试
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "test":
    test_c_model_integration()