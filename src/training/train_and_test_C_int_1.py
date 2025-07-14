#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
障碍物高度分类器 - 完整的整型C语言转换版本
包含LightGBM模型训练和自动C语言整型转换功能
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
import json
from datetime import datetime
import math

warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LightGBMToCIntegerConverter:
    """LightGBM模型转C语言整型代码生成器"""

    def __init__(self, model, feature_names, output_dir="./c_model_integer", scale_factor=10000):
        """
        初始化转换器

        Parameters:
        -----------
        model : LightGBM model
            训练好的LightGBM模型
        feature_names : list
            特征名称列表
        output_dir : str
            输出目录
        scale_factor : int
            定点数缩放因子，用于将浮点数转换为整数
            例如：scale_factor=10000表示精度为0.0001
        """
        print(f"🔍 初始化C语言整型转换器...")
        print(f"   - 模型类型: {type(model)}")
        print(f"   - 特征数量: {len(feature_names)}")
        print(f"   - 输出目录: {output_dir}")
        print(f"   - 缩放因子: {scale_factor} (精度: {1.0 / scale_factor})")

        self.model = model
        self.feature_names = feature_names
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.num_features = len(feature_names)
        self.num_trees = 0
        self.trees = []
        self.objective = 'binary'

        # 创建输出目录
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ 输出目录创建成功: {output_dir}")
        except Exception as e:
            print(f"❌ 输出目录创建失败: {e}")
            raise

    def float_to_fixed(self, value):
        """将浮点数转换为定点数整型"""
        return int(round(value * self.scale_factor))

    def extract_model_info(self):
        """提取模型信息"""
        print(f"🔍 提取模型信息...")

        try:
            if not hasattr(self.model, 'dump_model'):
                raise ValueError(f"模型类型 {type(self.model)} 不支持dump_model方法")

            model_dict = self.model.dump_model()
            print(f"✅ 模型信息提取成功")

            self.num_trees = model_dict.get('num_tree_per_iteration', 1)
            self.trees = model_dict.get('tree_info', [])
            self.objective = model_dict.get('objective', 'binary')

            print(f"📊 模型详细信息:")
            print(f"   - 树的数量: {len(self.trees)}")
            print(f"   - 每次迭代树数: {self.num_trees}")
            print(f"   - 特征数量: {self.num_features}")
            print(f"   - 目标函数: {self.objective}")

            return model_dict

        except Exception as e:
            print(f"❌ 模型信息提取失败: {e}")
            raise

    def generate_tree_function(self, tree_dict, tree_id):
        """生成单个树的C函数（整型版本）"""
        print(f"🌳 生成第 {tree_id} 棵树（整型版本）...")

        try:
            def generate_node_code(node, indent=0):
                """递归生成节点代码（整型版本）"""
                spaces = "    " * indent

                if 'leaf_value' in node:
                    leaf_value = node['leaf_value']
                    leaf_value_fixed = self.float_to_fixed(leaf_value)
                    return f"{spaces}return {leaf_value_fixed};\n"
                else:
                    feature_idx = node['split_feature']
                    threshold = node['threshold']
                    threshold_fixed = self.float_to_fixed(threshold)
                    left_child = node['left_child']
                    right_child = node['right_child']

                    if feature_idx >= self.num_features:
                        raise ValueError(f"特征索引 {feature_idx} 超出范围 [0, {self.num_features - 1}]")

                    code = f"{spaces}if (features[{feature_idx}] <= {threshold_fixed}) {{\n"
                    code += generate_node_code(left_child, indent + 1)
                    code += f"{spaces}}} else {{\n"
                    code += generate_node_code(right_child, indent + 1)
                    code += f"{spaces}}}\n"

                    return code

            if 'tree_structure' not in tree_dict:
                raise ValueError(f"树 {tree_id} 缺少 tree_structure")

            tree_structure = tree_dict['tree_structure']

            func_code = f"""
static int32_t tree_{tree_id}(const int32_t *features) {{
{generate_node_code(tree_structure, 1)}}}
"""
            print(f"✅ 第 {tree_id} 棵树生成成功（整型版本）")
            return func_code

        except Exception as e:
            print(f"❌ 第 {tree_id} 棵树生成失败: {e}")
            raise

    def generate_prediction_function(self):
        """生成预测函数（整型版本）"""
        print(f"🔮 生成预测函数（整型版本）...")

        try:
            tree_calls = []
            for i in range(len(self.trees)):
                tree_calls.append(f"    sum += tree_{i}(features);")

            tree_calls_str = "\n".join(tree_calls)

            # 生成整型版本的sigmoid查找表
            sigmoid_table_size = 1000
            sigmoid_table = []
            for i in range(sigmoid_table_size + 1):
                x = (i - sigmoid_table_size // 2) * 20.0 / sigmoid_table_size  # 范围大约是-10到10
                sigmoid_val = 1.0 / (1.0 + math.exp(-x))
                sigmoid_table.append(self.float_to_fixed(sigmoid_val))

            sigmoid_table_str = ",\n    ".join([str(val) for val in sigmoid_table])

            prediction_code = f"""
// Sigmoid查找表 (定点数格式，缩放因子: {self.scale_factor})
static const int32_t sigmoid_table[{sigmoid_table_size + 1}] = {{
    {sigmoid_table_str}
}};

#define SIGMOID_TABLE_SIZE {sigmoid_table_size}
#define SIGMOID_INPUT_SCALE 20
#define SIGMOID_INPUT_OFFSET {sigmoid_table_size // 2}

int32_t predict_raw(const int32_t *features) {{
    int64_t sum = 0;  // 使用64位避免溢出
{tree_calls_str}
    return (int32_t)sum;
}}

int32_t predict_probability_fixed(const int32_t *features) {{
    int32_t raw_score = predict_raw(features);

    // 将原始分数映射到查找表索引
    // raw_score是定点数格式，需要转换到合适的范围
    int32_t table_input = (raw_score * SIGMOID_INPUT_SCALE) / {self.scale_factor} + SIGMOID_INPUT_OFFSET;

    // 边界检查
    if (table_input < 0) table_input = 0;
    if (table_input > SIGMOID_TABLE_SIZE) table_input = SIGMOID_TABLE_SIZE;

    return sigmoid_table[table_input];
}}

int predict_class(const int32_t *features) {{
    int32_t prob = predict_probability_fixed(features);
    return prob > {self.scale_factor // 2} ? 1 : 0;  // 阈值0.5对应的定点数
}}

int32_t predict_confidence_fixed(const int32_t *features) {{
    int32_t prob = predict_probability_fixed(features);
    int32_t half_scale = {self.scale_factor // 2};
    int32_t diff = prob > half_scale ? (prob - half_scale) : (half_scale - prob);
    return diff * 2;  // 转换为0-1的置信度
}}

// 浮点数版本的接口（用于调试和验证）
double predict_probability_float(const int32_t *features) {{
    return (double)predict_probability_fixed(features) / {self.scale_factor};
}}

double predict_confidence_float(const int32_t *features) {{
    return (double)predict_confidence_fixed(features) / {self.scale_factor};
}}

double predict_raw_float(const int32_t *features) {{
    return (double)predict_raw(features) / {self.scale_factor};
}}
"""
            print(f"✅ 预测函数生成成功（整型版本）")
            return prediction_code

        except Exception as e:
            print(f"❌ 预测函数生成失败: {e}")
            raise

    def write_file(self, filepath, content):
        """写入文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✅ 文件写入成功: {os.path.basename(filepath)} ({file_size} bytes)")
                return True
            else:
                print(f"❌ 文件写入失败: {os.path.basename(filepath)} (文件不存在)")
                return False

        except Exception as e:
            print(f"❌ 文件写入失败: {os.path.basename(filepath)}")
            print(f"   错误: {e}")
            return False

    def convert(self):
        """执行转换"""
        print(f"\n🚀 开始转换LightGBM模型到C语言（整型版本）...")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"🔢 缩放因子: {self.scale_factor}")
        print(f"{'=' * 60}")

        try:
            # 检查输出目录权限
            if not os.path.exists(self.output_dir):
                print(f"❌ 输出目录不存在: {self.output_dir}")
                return False

            if not os.access(self.output_dir, os.W_OK):
                print(f"❌ 输出目录无写入权限: {self.output_dir}")
                return False

            print(f"✅ 输出目录检查通过")

            # 提取模型信息
            self.extract_model_info()

            # 生成头文件
            print(f"\n1️⃣ 生成头文件...")
            header_content = self.generate_header_file()
            header_file = os.path.join(self.output_dir, "obstacle_height_model_int.h")
            if not self.write_file(header_file, header_content):
                return False

            # 生成源文件
            print(f"\n2️⃣ 生成源文件...")
            source_content = self.generate_source_file()
            source_file = os.path.join(self.output_dir, "obstacle_height_model_int.c")
            if not self.write_file(source_file, source_content):
                return False

            # 生成测试文件
            print(f"\n3️⃣ 生成测试文件...")
            test_content = self.generate_test_file()
            test_file = os.path.join(self.output_dir, "test_obstacle_height_int.c")
            if not self.write_file(test_file, test_content):
                return False

            # 生成Makefile
            print(f"\n4️⃣ 生成Makefile...")
            makefile_content = self.generate_makefile()
            makefile_file = os.path.join(self.output_dir, "Makefile")
            if not self.write_file(makefile_file, makefile_content):
                return False

            # 保存特征映射
            print(f"\n5️⃣ 保存特征映射...")
            if not self.save_feature_mapping():
                return False

            # 生成数据转换工具
            print(f"\n6️⃣ 生成数据转换工具...")
            if not self.generate_data_converter():
                return False

            # 生成使用说明
            print(f"\n7️⃣ 生成使用说明...")
            if not self.generate_usage_guide():
                return False

            print(f"\n🎉 整型版本转换完成!")
            return True

        except Exception as e:
            print(f"\n❌ 转换过程中发生错误: {e}")
            return False

    def generate_header_file(self):
        """生成头文件（整型版本）"""
        header_content = f"""#ifndef OBSTACLE_HEIGHT_MODEL_INT_H
#define OBSTACLE_HEIGHT_MODEL_INT_H

#include <stdint.h>

// 模型信息
#define NUM_FEATURES {self.num_features}
#define NUM_TREES {len(self.trees)}
#define SCALE_FACTOR {self.scale_factor}
#define PRECISION_DIGITS {len(str(self.scale_factor)) - 1}

// 特征索引定义
"""
        for i, feature_name in enumerate(self.feature_names):
            feature_macro = (feature_name.upper()
                             .replace(' ', '_')
                             .replace('-', '_')
                             .replace('(', '')
                             .replace(')', '')
                             .replace('.', '_')
                             .replace('/', '_'))
            header_content += f"#define FEATURE_{feature_macro} {i}\n"

        header_content += f"""
// 定点数转换宏
#define FLOAT_TO_FIXED(x) ((int32_t)((x) * SCALE_FACTOR))
#define FIXED_TO_FLOAT(x) ((double)(x) / SCALE_FACTOR)

// 整型预测函数
int32_t predict_raw(const int32_t *features);
int32_t predict_probability_fixed(const int32_t *features);
int predict_class(const int32_t *features);
int32_t predict_confidence_fixed(const int32_t *features);

// 浮点数接口（用于调试）
double predict_probability_float(const int32_t *features);
double predict_confidence_float(const int32_t *features);
double predict_raw_float(const int32_t *features);

// 数据转换辅助函数
void convert_float_features_to_fixed(const double *float_features, int32_t *fixed_features);

// 预测结果结构体（整型版本）
typedef struct {{
    int height_label;
    int32_t probability_fixed;
    int32_t confidence_fixed;
    int32_t raw_score_fixed;
    double probability_float;  // 调试用
    double confidence_float;   // 调试用
    double raw_score_float;    // 调试用
}} ObstacleHeightPredictionInt;

ObstacleHeightPredictionInt predict_obstacle_height_int(const int32_t *features);
ObstacleHeightPredictionInt predict_obstacle_height_from_float(const double *float_features);

#endif // OBSTACLE_HEIGHT_MODEL_INT_H
"""
        return header_content

    def generate_source_file(self):
        """生成源文件（整型版本）"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_content = f"""/*
 * 障碍物高度分类器 - LightGBM模型C语言整型实现
 * 生成时间: {timestamp}
 * 特征数量: {self.num_features}
 * 树的数量: {len(self.trees)}
 * 缩放因子: {self.scale_factor}
 * 精度: {1.0 / self.scale_factor}
 */

#include "obstacle_height_model_int.h"
#include <math.h>

"""

        # 生成所有树的函数
        print(f"🌳 开始生成 {len(self.trees)} 棵树（整型版本）...")
        for i, tree in enumerate(self.trees):
            tree_func = self.generate_tree_function(tree, i)
            source_content += tree_func

        # 生成预测函数
        prediction_func = self.generate_prediction_function()
        source_content += prediction_func

        # 生成便利函数
        source_content += f"""
void convert_float_features_to_fixed(const double *float_features, int32_t *fixed_features) {{
    for (int i = 0; i < NUM_FEATURES; i++) {{
        fixed_features[i] = FLOAT_TO_FIXED(float_features[i]);
    }}
}}

ObstacleHeightPredictionInt predict_obstacle_height_int(const int32_t *features) {{
    ObstacleHeightPredictionInt result;
    result.raw_score_fixed = predict_raw(features);
    result.probability_fixed = predict_probability_fixed(features);
    result.height_label = predict_class(features);
    result.confidence_fixed = predict_confidence_fixed(features);

    // 生成浮点数版本用于调试
    result.raw_score_float = FIXED_TO_FLOAT(result.raw_score_fixed);
    result.probability_float = FIXED_TO_FLOAT(result.probability_fixed);
    result.confidence_float = FIXED_TO_FLOAT(result.confidence_fixed);

    return result;
}}

ObstacleHeightPredictionInt predict_obstacle_height_from_float(const double *float_features) {{
    int32_t fixed_features[NUM_FEATURES];
    convert_float_features_to_fixed(float_features, fixed_features);
    return predict_obstacle_height_int(fixed_features);
}}
"""

        return source_content

    def generate_test_file(self):
        """生成测试文件（整型版本）"""
        test_content = f"""/*
 * 障碍物高度分类器测试程序（整型版本）
 */

#include <stdio.h>
#include <stdlib.h>
#include "obstacle_height_model_int.h"

void test_integer_prediction() {{
    printf("=== 整型预测测试 ===\\n");

    // 示例测试数据（整型定点数格式）
    int32_t fixed_features[NUM_FEATURES] = {{0}};

    // 设置一些示例值（已经是定点数格式）
    for(int i = 0; i < NUM_FEATURES && i < 10; i++) {{
        fixed_features[i] = FLOAT_TO_FIXED(i * 100 + 200);
    }}

    printf("输入特征（前10个）:\\n");
    for(int i = 0; i < NUM_FEATURES && i < 10; i++) {{
        printf("特征%d: %d (%.6f)\\n", i, fixed_features[i], FIXED_TO_FLOAT(fixed_features[i]));
    }}

    ObstacleHeightPredictionInt result = predict_obstacle_height_int(fixed_features);

    printf("\\n整型预测结果:\\n");
    printf("- 高度标签: %d (%s)\\n", result.height_label, 
           result.height_label ? "高障碍物" : "低障碍物");
    printf("- 概率(定点数): %d\\n", result.probability_fixed);
    printf("- 概率(浮点数): %.6f\\n", result.probability_float);
    printf("- 置信度(定点数): %d\\n", result.confidence_fixed);
    printf("- 置信度(浮点数): %.6f\\n", result.confidence_float);
    printf("- 原始分数(定点数): %d\\n", result.raw_score_fixed);
    printf("- 原始分数(浮点数): %.6f\\n", result.raw_score_float);
}}

void test_float_to_fixed_conversion() {{
    printf("\\n=== 浮点数转换测试 ===\\n");

    // 测试浮点数特征
    double float_features[NUM_FEATURES] = {{0.0}};

    // 设置一些示例值
    for(int i = 0; i < NUM_FEATURES && i < 10; i++) {{
        float_features[i] = (double)(i * 100 + 200);
    }}

    printf("浮点数输入特征（前10个）:\\n");
    for(int i = 0; i < NUM_FEATURES && i < 10; i++) {{
        printf("特征%d: %.6f\\n", i, float_features[i]);
    }}

    ObstacleHeightPredictionInt result = predict_obstacle_height_from_float(float_features);

    printf("\\n从浮点数转换后的预测结果:\\n");
    printf("- 高度标签: %d (%s)\\n", result.height_label, 
           result.height_label ? "高障碍物" : "低障碍物");
    printf("- 概率: %.6f\\n", result.probability_float);
    printf("- 置信度: %.6f\\n", result.confidence_float);
    printf("- 原始分数: %.6f\\n", result.raw_score_float);
}}

void test_precision() {{
    printf("\\n=== 精度测试 ===\\n");
    printf("缩放因子: %d\\n", SCALE_FACTOR);
    printf("理论精度: %.8f\\n", 1.0 / SCALE_FACTOR);

    // 测试一些关键值的转换精度
    double test_values[] = {{0.0, 0.5, 1.0, 0.1, 0.9, 0.123456}};
    int num_test_values = sizeof(test_values) / sizeof(test_values[0]);

    printf("\\n转换精度测试:\\n");
    for(int i = 0; i < num_test_values; i++) {{
        double original = test_values[i];
        int32_t fixed = FLOAT_TO_FIXED(original);
        double recovered = FIXED_TO_FLOAT(fixed);
        double error = fabs(original - recovered);

        printf("原值: %.6f -> 定点数: %d -> 恢复值: %.6f, 误差: %.8f\\n",
               original, fixed, recovered, error);
    }}
}}

int main() {{
    printf("=== 障碍物高度分类器 C语言整型版本 ===\\n");
    printf("特征数量: %d\\n", NUM_FEATURES);
    printf("树的数量: %d\\n", NUM_TREES);
    printf("缩放因子: %d\\n", SCALE_FACTOR);
    printf("\\n");

    test_precision();
    test_integer_prediction();
    test_float_to_fixed_conversion();

    return 0;
}}
"""
        return test_content

    def generate_makefile(self):
        """生成Makefile（整型版本）"""
        return """# 障碍物高度分类器 Makefile (整型版本)

CC = gcc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

TARGET = test_obstacle_height_int
SOURCES = obstacle_height_model_int.c test_obstacle_height_int.c
HEADERS = obstacle_height_model_int.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
"""

    def save_feature_mapping(self):
        """保存特征映射（整型版本）"""
        try:
            feature_mapping = {
                "feature_names": self.feature_names,
                "feature_count": self.num_features,
                "scale_factor": self.scale_factor,
                "precision": 1.0 / self.scale_factor,
                "model_info": {
                    "trees": len(self.trees),
                    "objective": self.objective,
                    "data_type": "int32_t"
                }
            }

            mapping_file = os.path.join(self.output_dir, "feature_mapping_int.json")
            return self.write_file(mapping_file, json.dumps(feature_mapping, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"❌ 特征映射保存失败: {e}")
            return False

    def generate_data_converter(self):
        """生成数据转换工具"""
        try:
            converter_content = f"""/*
 * 数据转换工具 - 浮点数到定点数转换
 * 缩放因子: {self.scale_factor}
 */

#include <stdio.h>
#include <stdlib.h>
#include "obstacle_height_model_int.h"

// 批量转换浮点数特征到定点数
void batch_convert_features(const double *float_batch, int32_t *fixed_batch, int num_samples) {{
    for (int sample = 0; sample < num_samples; sample++) {{
        for (int feature = 0; feature < NUM_FEATURES; feature++) {{
            int input_idx = sample * NUM_FEATURES + feature;
            int output_idx = sample * NUM_FEATURES + feature;
            fixed_batch[output_idx] = FLOAT_TO_FIXED(float_batch[input_idx]);
        }}
    }}
}}

// 从CSV文件读取并转换数据（示例函数）
int convert_csv_to_fixed(const char *input_csv, const char *output_file) {{
    FILE *input = fopen(input_csv, "r");
    FILE *output = fopen(output_file, "w");

    if (!input || !output) {{
        printf("文件打开失败\\n");
        return -1;
    }}

    char line[4096];
    int line_num = 0;

    // 写入头部信息
    fprintf(output, "# 转换后的定点数特征文件\\n");
    fprintf(output, "# 缩放因子: %d\\n", SCALE_FACTOR);
    fprintf(output, "# 特征数量: %d\\n", NUM_FEATURES);
    fprintf(output, "\\n");

    while (fgets(line, sizeof(line), input)) {{
        line_num++;
        if (line_num == 1) continue; // 跳过标题行

        double features[NUM_FEATURES];
        int32_t fixed_features[NUM_FEATURES];

        // 解析CSV行（简化版本，实际使用时可能需要更复杂的解析）
        char *token = strtok(line, ",");
        int feature_idx = 0;

        while (token && feature_idx < NUM_FEATURES) {{
            features[feature_idx] = atof(token);
            feature_idx++;
            token = strtok(NULL, ",");
        }}

        // 转换为定点数
        convert_float_features_to_fixed(features, fixed_features);

        // 写入输出文件
        for (int i = 0; i < NUM_FEATURES; i++) {{
            fprintf(output, "%d", fixed_features[i]);
            if (i < NUM_FEATURES - 1) fprintf(output, ",");
        }}
        fprintf(output, "\\n");
    }}

    fclose(input);
    fclose(output);

    printf("转换完成，处理了 %d 行数据\\n", line_num - 1);
    return 0;
}}

int main(int argc, char *argv[]) {{
    if (argc < 3) {{
        printf("用法: %s <输入CSV文件> <输出文件>\\n", argv[0]);
        return 1;
    }}

    return convert_csv_to_fixed(argv[1], argv[2]);
}}
"""

            converter_file = os.path.join(self.output_dir, "data_converter.c")
            return self.write_file(converter_file, converter_content)

        except Exception as e:
            print(f"❌ 数据转换工具生成失败: {e}")
            return False

    def generate_usage_guide(self):
        """生成使用说明（整型版本）"""
        try:
            guide_content = f"""# 障碍物高度分类器 C语言整型版本

## 概述
本版本专为MCU环境设计，使用32位整型进行所有计算，避免浮点运算。

## 技术参数
- **数据类型**: int32_t (32位有符号整数)
- **缩放因子**: {self.scale_factor}
- **数值精度**: {1.0 / self.scale_factor}
- **特征数量**: {self.num_features}
- **树的数量**: {len(self.trees)}

## 编译和运行
```bash
# 编译测试程序
make

# 运行测试
./test_obstacle_height_int

# 编译数据转换工具
gcc -o data_converter data_converter.c obstacle_height_model_int.c -lm
```

## 在MCU项目中使用

### 1. 基本使用
```c
#include "obstacle_height_model_int.h"

// 定义特征数组（定点数格式）
int32_t features[NUM_FEATURES];

// 设置特征值（需要转换为定点数）
features[0] = FLOAT_TO_FIXED(123.45);  // 将浮点数转换为定点数
features[1] = FLOAT_TO_FIXED(67.89);

// 进行预测
ObstacleHeightPredictionInt result = predict_obstacle_height_int(features);

if (result.height_label == 1) {{
    // 检测到高障碍物
    printf("高障碍物检测！置信度: %d\\n", result.confidence_fixed);
}}
```

### 2. 如果输入是浮点数
```c
double float_features[NUM_FEATURES] = {{123.45, 67.89, ...}};
ObstacleHeightPredictionInt result = predict_obstacle_height_from_float(float_features);
```

### 3. 手动数据转换
```c
// 浮点数转定点数
double float_value = 123.456;
int32_t fixed_value = FLOAT_TO_FIXED(float_value);

// 定点数转浮点数（用于调试）
double recovered_value = FIXED_TO_FLOAT(fixed_value);
```

## 内存使用
- **特征数组**: {self.num_features} × 4 = {self.num_features * 4} 字节
- **预测结果**: 约 32 字节
- **sigmoid查找表**: 约 4KB
- **代码大小**: 预计 < 50KB

## 性能特点
- ✅ 无浮点运算，适合无FPU的MCU
- ✅ 查找表实现sigmoid函数，速度快
- ✅ 32位整型，精度足够
- ✅ 内存占用小

## 精度分析
使用 {self.scale_factor} 作为缩放因子：
- 最小表示值: {1.0 / self.scale_factor}
- 数值范围: ±{2 ** 31 // self.scale_factor}
- 对于障碍物分类任务，此精度通常足够

## 特征列表 ({self.num_features}个特征)
"""
            for i, name in enumerate(self.feature_names):
                guide_content += f"{i:2d}: {name}\n"

            guide_content += f"""

## API 参考

### 核心预测函数
```c
// 原始分数预测（定点数）
int32_t predict_raw(const int32_t *features);

// 概率预测（定点数，0-{self.scale_factor}表示0-1）
int32_t predict_probability_fixed(const int32_t *features);

// 分类预测（0或1）
int predict_class(const int32_t *features);

// 置信度预测（定点数）
int32_t predict_confidence_fixed(const int32_t *features);
```

### 辅助函数
```c
// 批量特征转换
void convert_float_features_to_fixed(const double *float_features, int32_t *fixed_features);

// 完整预测（推荐使用）
ObstacleHeightPredictionInt predict_obstacle_height_int(const int32_t *features);
ObstacleHeightPredictionInt predict_obstacle_height_from_float(const double *float_features);
```

## 移植到MCU注意事项

1. **包含文件**: 确保 `stdint.h` 可用
2. **数学库**: 可能需要移除 `math.h` 依赖
3. **内存对齐**: 注意32位整数的内存对齐
4. **查找表**: sigmoid_table占用约4KB ROM

## 调试技巧

1. **精度验证**:
   ```c
   test_precision(); // 运行精度测试
   ```

2. **结果对比**:
   ```c
   // 同时查看定点数和浮点数结果
   printf("定点数概率: %d, 浮点数概率: %.6f\\n", 
          result.probability_fixed, result.probability_float);
   ```

3. **特征检查**:
   ```c
   // 检查特征转换是否正确
   for(int i = 0; i < NUM_FEATURES; i++) {{
       printf("特征%d: %d (%.6f)\\n", i, fixed_features[i], 
              FIXED_TO_FLOAT(fixed_features[i]));
   }}
   ```

## 常见问题

**Q: 为什么使用定点数？**
A: 许多MCU没有浮点运算单元(FPU)，定点数运算更快更节能。

**Q: 精度够用吗？**
A: 对于分类任务，{1.0 / self.scale_factor}的精度通常足够。可以通过调整SCALE_FACTOR来平衡精度和范围。

**Q: 如何选择缩放因子？**
A: 考虑数据范围和精度需求。当前设置可表示±{2 ** 31 // self.scale_factor}范围的数值。

**Q: 内存不够怎么办？**
A: 可以减小sigmoid查找表大小，或使用更简单的激活函数。
"""

            guide_file = os.path.join(self.output_dir, "README_INTEGER.md")
            return self.write_file(guide_file, guide_content)

        except Exception as e:
            print(f"❌ 使用说明生成失败: {e}")
            return False


class ObstacleHeightClassifier:
    """障碍物高度分类器 - 基础版本"""

    def __init__(self, config=None):
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
            'save_model': True,
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
            self.create_directories()

    def create_directories(self):
        """创建输出目录"""
        dirs = [
            self.config['model_dir'],
            self.config['plot_dir'],
            self.config['results_dir'],
            self.config['misclassified_dir']
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"确保目录存在: {dir_path}")

    def save_csv(self, df, filepath, encoding=None):
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
        """特征工程"""
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
        self.check_feature_quality(df_processed)

        return df_processed

    def check_feature_quality(self, df):
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
            self.evaluate_model()

        # 保存模型
        if self.config['save_model']:
            self.save_model()

        return self.test_results if X_test is not None else None

    def evaluate_model(self):
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
            return None

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

        self.save_csv(results, output_filepath)

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

    def load_model(self, filepath):
        """加载模型"""
        try:
            model_data = joblib.load(filepath)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
                if 'config' in model_data:
                    self.config.update(model_data['config'])
            else:
                self.model = model_data

            print(f"模型加载成功: {filepath}")
            print(f"特征数量: {len(self.feature_names)}")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False


class ObstacleHeightClassifierWithIntegerConversion(ObstacleHeightClassifier):
    """障碍物高度分类器 - 增加整型转换功能"""

    def __init__(self, config=None):
        super().__init__(config)

        # 添加整型转换相关配置
        self.config.update({
            'c_model_integer_dir': './c_model_integer',
            'auto_convert_to_c_integer': True,
            'scale_factor': 10000,  # 定点数缩放因子
        })

        if config:
            self.config.update(config)

    def convert_to_c_integer(self):
        """将训练好的模型转换为C语言整型代码"""
        if self.model is None:
            print("❌ 模型未训练，无法转换")
            return False

        try:
            print(f"📝 开始转换LightGBM模型为C语言整型代码...")
            print(f"📊 模型信息检查:")
            print(f"   - 模型类型: {type(self.model)}")
            print(f"   - 特征数量: {len(self.feature_names)}")
            print(f"   - 缩放因子: {self.config['scale_factor']}")
            print(f"   - 输出目录: {self.config['c_model_integer_dir']}")

            # 检查模型是否有必要的方法
            if not hasattr(self.model, 'dump_model'):
                print(f"❌ 模型不支持dump_model方法")
                return False

            # 检查特征名称
            if not self.feature_names:
                print(f"❌ 特征名称列表为空")
                return False

            print(f"✅ 模型检查通过，开始转换...")

            # 使用整型转换器
            converter = LightGBMToCIntegerConverter(
                model=self.model,
                feature_names=self.feature_names,
                output_dir=self.config['c_model_integer_dir'],
                scale_factor=self.config['scale_factor']
            )

            # 执行转换
            success = converter.convert()

            if success:
                print(f"\n🎉 C语言整型转换完成!")
                print(f"📁 C语言整型代码保存在: {self.config['c_model_integer_dir']}")
                print(f"🔨 编译命令: cd {self.config['c_model_integer_dir']} && make")
                print(f"🚀 运行测试: cd {self.config['c_model_integer_dir']} && ./test_obstacle_height_int")

                # 验证文件是否真的生成了
                expected_files = [
                    "obstacle_height_model_int.h",
                    "obstacle_height_model_int.c",
                    "test_obstacle_height_int.c",
                    "Makefile",
                    "data_converter.c"
                ]
                missing_files = []

                for filename in expected_files:
                    filepath = os.path.join(self.config['c_model_integer_dir'], filename)
                    if not os.path.exists(filepath):
                        missing_files.append(filename)

                if missing_files:
                    print(f"⚠️  警告: 以下文件未生成: {missing_files}")
                    return False
                else:
                    print(f"✅ 所有必要文件已生成")
                    return True
            else:
                print(f"❌ C语言整型转换失败")
                return False

        except Exception as e:
            print(f"❌ C语言整型转换异常: {e}")
            return False

    def train(self, train_df):
        """训练模型并自动转换为整型C代码"""
        # 调用父类的训练方法
        result = super().train(train_df)

        # 自动转换为整型C语言
        if self.config['auto_convert_to_c_integer']:
            print(f"\n🔄 开始自动转换为C语言整型代码...")
            self.convert_to_c_integer()

        return result

    def create_directories(self):
        """创建输出目录（包括整型版本目录）"""
        super().create_directories()
        os.makedirs(self.config['c_model_integer_dir'], exist_ok=True)
        print(f"确保目录存在: {self.config['c_model_integer_dir']}")


def quick_run_with_integer_conversion(train_file, test_file=None, config=None):
    """快速运行函数 - 含自动整型C语言转换"""
    print("=== 障碍物高度分类器 - 整型C语言版本 ===\n")

    # 默认配置
    default_config = {
        'model_dir': './models_integer',
        'plot_dir': './plots_integer',
        'results_dir': './results_integer',
        'misclassified_dir': './misclassified_integer',
        'c_model_integer_dir': './c_model_integer',
        'auto_convert_to_c_integer': True,
        'scale_factor': 10000,  # 可以调整精度
        'test_size': 0.0,  # 使用全部数据训练
        'random_state': 42,
    }

    if config:
        default_config.update(config)

    # 创建分类器
    classifier = ObstacleHeightClassifierWithIntegerConversion(default_config)

    # 训练
    print("1. 数据加载和特征工程")
    train_df = classifier.load_data(train_file)
    train_df_processed = classifier.feature_engineering(train_df)

    print("\n2. 模型训练和整型转换")
    classifier.train(train_df_processed)

    # 特征重要性分析
    print("\n3. 特征重要性分析")
    classifier.plot_feature_importance()

    # 预测测试数据
    if test_file and os.path.exists(test_file):
        print("\n4. 测试数据预测")
        classifier.predict(test_file)
    else:
        print("\n4. 跳过预测阶段（无测试文件）")

    print("\n=== 运行完成 ===")
    print(f"📊 Python模型文件: {classifier.config['model_dir']}")
    print(f"🔧 C语言整型代码: {classifier.config['c_model_integer_dir']}")

    return classifier


def convert_existing_model_to_c_integer(model_path, output_dir="./c_model_integer_converted", scale_factor=10000):
    """转换已保存的.joblib模型文件为C语言整型代码"""
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
        print(f"🔢 缩放因子: {scale_factor}")

        # 创建转换器并执行转换
        converter = LightGBMToCIntegerConverter(model, feature_names, output_dir, scale_factor)
        success = converter.convert()

        return success

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False


if __name__ == "__main__":
    # ==================== 配置区域 ====================

    # 文件路径配置
    TRAIN_FILE = r'D:\PythonProject\data\processed_data\merged_train_data_fixed.csv'
    TEST_FILE = r'D:\PythonProject\data\processed_data\train_group2.csv'

    # 输出路径配置
    OUTPUT_CONFIG = {
        'model_dir': r'D:\PythonProject\model\saved_model_integer',
        'plot_dir': r'D:\PythonProject\results\visualization_results_integer',
        'results_dir': r'D:\PythonProject\results\prediction_results_integer',
        'misclassified_dir': r'D:\PythonProject\results\misclassified_results_integer',
        'c_model_integer_dir': r'D:\PythonProject\c_model_integer',  # 整型C语言模型输出目录
        'auto_create_dirs': True,
        'encoding': 'utf-8-sig',
        'test_size': 0.0,  # 使用全部数据训练
        'random_state': 42,
        'save_model': True,
        'auto_convert_to_c_integer': True,  # 启用自动转换为整型C语言
        'scale_factor': 10000,  # 定点数缩放因子，可根据需要调整
    }

    print(f"🚀 开始运行整型C语言版障碍物分类器...")
    print(f"📁 整型C语言模型将保存在: {OUTPUT_CONFIG['c_model_integer_dir']}")
    print(f"🔢 缩放因子: {OUTPUT_CONFIG['scale_factor']} (精度: {1.0 / OUTPUT_CONFIG['scale_factor']})")
    print(f"{'=' * 80}")

    try:
        # 检查训练文件是否存在
        if not os.path.exists(TRAIN_FILE):
            print(f"❌ 训练文件不存在: {TRAIN_FILE}")
            print(f"请检查文件路径是否正确")
            exit(1)

        # 运行分类器
        classifier = quick_run_with_integer_conversion(TRAIN_FILE, TEST_FILE, OUTPUT_CONFIG)

        print(f"\n{'=' * 80}")
        print(f"✅ 训练和整型转换完成！")
        print(f"\n📊 Python模型和结果:")
        print(f"   - 模型文件: {OUTPUT_CONFIG['model_dir']}")
        print(f"   - 可视化结果: {OUTPUT_CONFIG['plot_dir']}")
        print(f"   - 预测结果: {OUTPUT_CONFIG['results_dir']}")

        print(f"\n🔧 C语言整型模型:")
        print(f"   - C代码目录: {OUTPUT_CONFIG['c_model_integer_dir']}")
        print(f"   - 编译命令: cd {OUTPUT_CONFIG['c_model_integer_dir']} && make")
        print(f"   - 运行测试: cd {OUTPUT_CONFIG['c_model_integer_dir']} && ./test_obstacle_height_int")

        # 验证C语言文件是否生成
        c_files = [
            'obstacle_height_model_int.h',
            'obstacle_height_model_int.c',
            'test_obstacle_height_int.c',
            'Makefile',
            'data_converter.c'
        ]
        missing_files = []

        for filename in c_files:
            filepath = os.path.join(OUTPUT_CONFIG['c_model_integer_dir'], filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
            else:
                size = os.path.getsize(filepath)
                print(f"   ✅ {filename} ({size} bytes)")

        if missing_files:
            print(f"\n⚠️  警告: 以下C语言文件未生成: {missing_files}")
        else:
            print(f"\n🎉 所有C语言整型文件已成功生成!")
            print(f"\n📝 整型版本特点:")
            print(f"   - 使用 int32_t 数据类型")
            print(f"   - 缩放因子: {OUTPUT_CONFIG['scale_factor']}")
            print(f"   - 数值精度: {1.0 / OUTPUT_CONFIG['scale_factor']}")
            print(f"   - 适合无FPU的MCU")
            print(f"   - 查找表实现sigmoid函数")

            print(f"\n📝 下一步:")
            print(f"   1. 进入C模型目录: cd {OUTPUT_CONFIG['c_model_integer_dir']}")
            print(f"   2. 编译项目: make")
            print(f"   3. 运行测试: ./test_obstacle_height_int")
            print(f"   4. 查看README_INTEGER.md了解详细使用方法")

    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print(f"🔧 请检查:")
        print(f"   1. 训练数据文件是否存在: {TRAIN_FILE}")
        print(f"   2. 测试数据文件是否存在: {TEST_FILE}")

    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print(f"\n🔧 故障排除建议:")
        print(f"   1. 检查Python环境和依赖包")
        print(f"   2. 检查数据文件格式和内容")
        print(f"   3. 检查输出目录权限")
        print(f"   4. 尝试调整缩放因子scale_factor")

    print(f"\n🏁 程序结束")