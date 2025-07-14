/*
 * 障碍物高度分类器测试程序（整型版本）
 */

#include <stdio.h>
#include <stdlib.h>
#include "obstacle_height_model_int.h"

void test_integer_prediction() {
    printf("=== 整型预测测试 ===\n");

    // 示例测试数据（整型定点数格式）
    int32_t fixed_features[NUM_FEATURES] = {0};

    // 设置一些示例值（已经是定点数格式）
    for(int i = 0; i < NUM_FEATURES && i < 10; i++) {
        fixed_features[i] = FLOAT_TO_FIXED(i * 100 + 200);
    }

    printf("输入特征（前10个）:\n");
    for(int i = 0; i < NUM_FEATURES && i < 10; i++) {
        printf("特征%d: %d (%.6f)\n", i, fixed_features[i], FIXED_TO_FLOAT(fixed_features[i]));
    }

    ObstacleHeightPredictionInt result = predict_obstacle_height_int(fixed_features);

    printf("\n整型预测结果:\n");
    printf("- 高度标签: %d (%s)\n", result.height_label, 
           result.height_label ? "高障碍物" : "低障碍物");
    printf("- 概率(定点数): %d\n", result.probability_fixed);
    printf("- 概率(浮点数): %.6f\n", result.probability_float);
    printf("- 置信度(定点数): %d\n", result.confidence_fixed);
    printf("- 置信度(浮点数): %.6f\n", result.confidence_float);
    printf("- 原始分数(定点数): %d\n", result.raw_score_fixed);
    printf("- 原始分数(浮点数): %.6f\n", result.raw_score_float);
}

void test_float_to_fixed_conversion() {
    printf("\n=== 浮点数转换测试 ===\n");

    // 测试浮点数特征
    double float_features[NUM_FEATURES] = {0.0};

    // 设置一些示例值
    for(int i = 0; i < NUM_FEATURES && i < 10; i++) {
        float_features[i] = (double)(i * 100 + 200);
    }

    printf("浮点数输入特征（前10个）:\n");
    for(int i = 0; i < NUM_FEATURES && i < 10; i++) {
        printf("特征%d: %.6f\n", i, float_features[i]);
    }

    ObstacleHeightPredictionInt result = predict_obstacle_height_from_float(float_features);

    printf("\n从浮点数转换后的预测结果:\n");
    printf("- 高度标签: %d (%s)\n", result.height_label, 
           result.height_label ? "高障碍物" : "低障碍物");
    printf("- 概率: %.6f\n", result.probability_float);
    printf("- 置信度: %.6f\n", result.confidence_float);
    printf("- 原始分数: %.6f\n", result.raw_score_float);
}

void test_precision() {
    printf("\n=== 精度测试 ===\n");
    printf("缩放因子: %d\n", SCALE_FACTOR);
    printf("理论精度: %.8f\n", 1.0 / SCALE_FACTOR);

    // 测试一些关键值的转换精度
    double test_values[] = {0.0, 0.5, 1.0, 0.1, 0.9, 0.123456};
    int num_test_values = sizeof(test_values) / sizeof(test_values[0]);

    printf("\n转换精度测试:\n");
    for(int i = 0; i < num_test_values; i++) {
        double original = test_values[i];
        int32_t fixed = FLOAT_TO_FIXED(original);
        double recovered = FIXED_TO_FLOAT(fixed);
        double error = fabs(original - recovered);

        printf("原值: %.6f -> 定点数: %d -> 恢复值: %.6f, 误差: %.8f\n",
               original, fixed, recovered, error);
    }
}

int main() {
    printf("=== 障碍物高度分类器 C语言整型版本 ===\n");
    printf("特征数量: %d\n", NUM_FEATURES);
    printf("树的数量: %d\n", NUM_TREES);
    printf("缩放因子: %d\n", SCALE_FACTOR);
    printf("\n");

    test_precision();
    test_integer_prediction();
    test_float_to_fixed_conversion();

    return 0;
}
