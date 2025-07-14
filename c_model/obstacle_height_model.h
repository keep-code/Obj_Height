#ifndef OBSTACLE_HEIGHT_MODEL_H
#define OBSTACLE_HEIGHT_MODEL_H

#include <math.h>

// 模型信息
#define NUM_FEATURES 34
#define NUM_TREES 1000

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

// 函数声明
double predict_raw(const double *features);
double predict_probability(const double *features);
int predict_class(const double *features);
double predict_confidence(const double *features);

// 预测结果结构体
typedef struct {
    int height_label;
    double probability;
    double confidence;
    double raw_score;
} ObstacleHeightPrediction;

ObstacleHeightPrediction predict_obstacle_height(const double *features);

#endif // OBSTACLE_HEIGHT_MODEL_H
