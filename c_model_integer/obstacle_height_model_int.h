#ifndef OBSTACLE_HEIGHT_MODEL_INT_H
#define OBSTACLE_HEIGHT_MODEL_INT_H

#include <stdint.h>

// 模型信息
#define NUM_FEATURES 34
#define NUM_TREES 1000
#define SCALE_FACTOR 10000
#define PRECISION_DIGITS 4

// 特征索引定义
#define FEATURE_POSDEDIS1 0
#define FEATURE_POSDEAMP1 1
#define FEATURE_POSDEDIS2 2
#define FEATURE_POSDEAMP2 3
#define FEATURE_POSCEDIS1 4
#define FEATURE_POSCEAMP1 5
#define FEATURE_POSCEDIS2 6
#define FEATURE_POSCEAMP2 7
#define FEATURE_TRAINOBJDIST 8
#define FEATURE_AVGDEECHOHIGH_SAMETX 9
#define FEATURE_AVGCEECHOHIGH_SAMETXRX 10
#define FEATURE_ODODIFFOBJDIS 11
#define FEATURE_ODODIFFDEDIS 12
#define FEATURE_ODODIFF 13
#define FEATURE_OBJDIFF 14
#define FEATURE_COSANGLE 15
#define FEATURE_RATEOFVHODEDIFF 16
#define FEATURE_ANGLEDIST 17
#define FEATURE_DEECHO_RATIO 18
#define FEATURE_CEECHO_RATIO 19
#define FEATURE_DEAMP_RATIO 20
#define FEATURE_CEAMP_RATIO 21
#define FEATURE_TOTAL_DEECHO 22
#define FEATURE_TOTAL_CEECHO 23
#define FEATURE_TOTAL_DEAMP 24
#define FEATURE_TOTAL_CEAMP 25
#define FEATURE_DEDIS_DIFF 26
#define FEATURE_CEDIS_DIFF 27
#define FEATURE_DEAMP_DIFF 28
#define FEATURE_CEAMP_DIFF 29
#define FEATURE_AVG_ECHO_STRENGTH 30
#define FEATURE_DISTANCE_RATIO 31
#define FEATURE_ECHO_STRENGTH_RATIO 32
#define FEATURE_ODO_STABILITY 33

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
typedef struct {
    int height_label;
    int32_t probability_fixed;
    int32_t confidence_fixed;
    int32_t raw_score_fixed;
    double probability_float;  // 调试用
    double confidence_float;   // 调试用
    double raw_score_float;    // 调试用
} ObstacleHeightPredictionInt;

ObstacleHeightPredictionInt predict_obstacle_height_int(const int32_t *features);
ObstacleHeightPredictionInt predict_obstacle_height_from_float(const double *float_features);

#endif // OBSTACLE_HEIGHT_MODEL_INT_H
