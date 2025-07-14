# 障碍物高度分类器 C语言整型版本

## 概述
本版本专为MCU环境设计，使用32位整型进行所有计算，避免浮点运算。

## 技术参数
- **数据类型**: int32_t (32位有符号整数)
- **缩放因子**: 10000
- **数值精度**: 0.0001
- **特征数量**: 34
- **树的数量**: 1000

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

if (result.height_label == 1) {
    // 检测到高障碍物
    printf("高障碍物检测！置信度: %d\n", result.confidence_fixed);
}
```

### 2. 如果输入是浮点数
```c
double float_features[NUM_FEATURES] = {123.45, 67.89, ...};
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
- **特征数组**: 34 × 4 = 136 字节
- **预测结果**: 约 32 字节
- **sigmoid查找表**: 约 4KB
- **代码大小**: 预计 < 50KB

## 性能特点
- ✅ 无浮点运算，适合无FPU的MCU
- ✅ 查找表实现sigmoid函数，速度快
- ✅ 32位整型，精度足够
- ✅ 内存占用小

## 精度分析
使用 10000 作为缩放因子：
- 最小表示值: 0.0001
- 数值范围: ±214748
- 对于障碍物分类任务，此精度通常足够

## 特征列表 (34个特征)
 0: PosDeDis1
 1: PosDeAmp1
 2: PosDeDis2
 3: PosDeAmp2
 4: PosCeDis1
 5: PosCeAmp1
 6: PosCeDis2
 7: PosCeAmp2
 8: TrainObjDist
 9: AvgDeEchoHigh_SameTx
10: AvgCeEchoHigh_SameTxRx
11: OdoDiffObjDis
12: OdoDiffDeDis
13: OdoDiff
14: ObjDiff
15: CosAngle
16: RateOfVhoDeDiff
17: AngleDist
18: DeEcho_Ratio
19: CeEcho_Ratio
20: DeAmp_Ratio
21: CeAmp_Ratio
22: Total_DeEcho
23: Total_CeEcho
24: Total_DeAmp
25: Total_CeAmp
26: DeDis_Diff
27: CeDis_Diff
28: DeAmp_Diff
29: CeAmp_Diff
30: Avg_Echo_Strength
31: Distance_Ratio
32: Echo_Strength_Ratio
33: Odo_Stability


## API 参考

### 核心预测函数
```c
// 原始分数预测（定点数）
int32_t predict_raw(const int32_t *features);

// 概率预测（定点数，0-10000表示0-1）
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
   printf("定点数概率: %d, 浮点数概率: %.6f\n", 
          result.probability_fixed, result.probability_float);
   ```

3. **特征检查**:
   ```c
   // 检查特征转换是否正确
   for(int i = 0; i < NUM_FEATURES; i++) {
       printf("特征%d: %d (%.6f)\n", i, fixed_features[i], 
              FIXED_TO_FLOAT(fixed_features[i]));
   }
   ```

## 常见问题

**Q: 为什么使用定点数？**
A: 许多MCU没有浮点运算单元(FPU)，定点数运算更快更节能。

**Q: 精度够用吗？**
A: 对于分类任务，0.0001的精度通常足够。可以通过调整SCALE_FACTOR来平衡精度和范围。

**Q: 如何选择缩放因子？**
A: 考虑数据范围和精度需求。当前设置可表示±214748范围的数值。

**Q: 内存不够怎么办？**
A: 可以减小sigmoid查找表大小，或使用更简单的激活函数。
