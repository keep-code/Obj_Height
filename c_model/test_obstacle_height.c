/*
 * 障碍物高度分类器测试程序
 */

#include <stdio.h>
#include <stdlib.h>
#include "obstacle_height_model.h"

int main() {
    printf("=== 障碍物高度分类器 C语言版本 ===\n");
    printf("特征数量: %d\n", NUM_FEATURES);
    printf("树的数量: %d\n", NUM_TREES);
    printf("\n");

    // 示例测试
    double features[NUM_FEATURES] = {0};

    // 设置一些示例值
    for(int i = 0; i < NUM_FEATURES && i < 10; i++) {
        features[i] = (double)(i * 100 + 200);
    }

    ObstacleHeightPrediction result = predict_obstacle_height(features);

    printf("预测结果:\n");
    printf("- 高度标签: %d (%s)\n", result.height_label, 
           result.height_label ? "高障碍物" : "低障碍物");
    printf("- 概率: %.6f\n", result.probability);
    printf("- 置信度: %.6f\n", result.confidence);
    printf("- 原始分数: %.6f\n", result.raw_score);

    return 0;
}
