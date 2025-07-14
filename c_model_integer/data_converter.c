/*
 * 数据转换工具 - 浮点数到定点数转换
 * 缩放因子: 10000
 */

#include <stdio.h>
#include <stdlib.h>
#include "obstacle_height_model_int.h"

// 批量转换浮点数特征到定点数
void batch_convert_features(const double *float_batch, int32_t *fixed_batch, int num_samples) {
    for (int sample = 0; sample < num_samples; sample++) {
        for (int feature = 0; feature < NUM_FEATURES; feature++) {
            int input_idx = sample * NUM_FEATURES + feature;
            int output_idx = sample * NUM_FEATURES + feature;
            fixed_batch[output_idx] = FLOAT_TO_FIXED(float_batch[input_idx]);
        }
    }
}

// 从CSV文件读取并转换数据（示例函数）
int convert_csv_to_fixed(const char *input_csv, const char *output_file) {
    FILE *input = fopen(input_csv, "r");
    FILE *output = fopen(output_file, "w");

    if (!input || !output) {
        printf("文件打开失败\n");
        return -1;
    }

    char line[4096];
    int line_num = 0;

    // 写入头部信息
    fprintf(output, "# 转换后的定点数特征文件\n");
    fprintf(output, "# 缩放因子: %d\n", SCALE_FACTOR);
    fprintf(output, "# 特征数量: %d\n", NUM_FEATURES);
    fprintf(output, "\n");

    while (fgets(line, sizeof(line), input)) {
        line_num++;
        if (line_num == 1) continue; // 跳过标题行

        double features[NUM_FEATURES];
        int32_t fixed_features[NUM_FEATURES];

        // 解析CSV行（简化版本，实际使用时可能需要更复杂的解析）
        char *token = strtok(line, ",");
        int feature_idx = 0;

        while (token && feature_idx < NUM_FEATURES) {
            features[feature_idx] = atof(token);
            feature_idx++;
            token = strtok(NULL, ",");
        }

        // 转换为定点数
        convert_float_features_to_fixed(features, fixed_features);

        // 写入输出文件
        for (int i = 0; i < NUM_FEATURES; i++) {
            fprintf(output, "%d", fixed_features[i]);
            if (i < NUM_FEATURES - 1) fprintf(output, ",");
        }
        fprintf(output, "\n");
    }

    fclose(input);
    fclose(output);

    printf("转换完成，处理了 %d 行数据\n", line_num - 1);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("用法: %s <输入CSV文件> <输出文件>\n", argv[0]);
        return 1;
    }

    return convert_csv_to_fixed(argv[1], argv[2]);
}
