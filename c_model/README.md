# 障碍物高度分类器 C语言版本

## 编译和运行
```bash
make
./test_obstacle_height
```

## 在你的项目中使用
```c
#include "obstacle_height_model.h"

double features[NUM_FEATURES];
// 设置特征值...

ObstacleHeightPrediction result = predict_obstacle_height(features);
if (result.height_label == 1) {
    printf("高障碍物检测！\n");
}
```

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
