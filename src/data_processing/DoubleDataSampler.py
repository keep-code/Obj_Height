import pandas as pd
import numpy as np
from sklearn.utils import resample
import argparse
import os


class DataBalanceSampler:
    """
    数据平衡采样工具
    用于从不平衡数据集中随机采样指定数量的不同类别样本
    """

    def __init__(self, csv_file_path, target_column='HeightLabel'):
        """
        初始化采样器

        参数:
        csv_file_path: str, CSV文件路径
        target_column: str, 目标列名，默认为'HeightLabel'
        """
        self.csv_file_path = csv_file_path
        self.target_column = target_column
        self.df = None
        self.class_counts = None

    def load_data(self):
        """加载数据并显示基本信息"""
        print(f"正在加载数据: {self.csv_file_path}")

        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"文件不存在: {self.csv_file_path}")

        self.df = pd.read_csv(self.csv_file_path)
        print(f"数据形状: {self.df.shape}")

        # 检查目标列是否存在
        if self.target_column not in self.df.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不存在。可用列: {list(self.df.columns)}")

        # 统计各类别数量
        self.class_counts = self.df[self.target_column].value_counts().sort_index()
        print(f"\n当前数据分布:")
        for class_label, count in self.class_counts.items():
            print(f"  类别 {class_label}: {count} 条记录 ({count / len(self.df) * 100:.1f}%)")

        return self.df

    def dual_group_sample(self, group1_class_0, group1_class_1, group2_class_0, group2_class_1,
                          random_state=42, output_file1=None, output_file2=None):
        """
        双组不重复采样：从数据中采样两组数据，每组包含指定数量的类别0和类别1样本，
        两组之间不能有重复的样本

        参数:
        group1_class_0: int, 第一组类别0的样本数量
        group1_class_1: int, 第一组类别1的样本数量
        group2_class_0: int, 第二组类别0的样本数量
        group2_class_1: int, 第二组类别1的样本数量
        random_state: int, 随机种子
        output_file1: str, 第一组输出文件名
        output_file2: str, 第二组输出文件名

        返回:
        tuple: (group1_data, group2_data) 两组采样后的数据
        """
        if self.df is None:
            raise ValueError("请先使用 load_data() 加载数据")

        # 检查请求的样本数量是否超过可用数量
        available_0 = self.class_counts.get(0, 0)
        available_1 = self.class_counts.get(1, 0)

        total_needed_0 = group1_class_0 + group2_class_0
        total_needed_1 = group1_class_1 + group2_class_1

        if total_needed_0 > available_0:
            raise ValueError(f"请求的类别0总样本数({total_needed_0})超过可用数量({available_0})")

        if total_needed_1 > available_1:
            raise ValueError(f"请求的类别1总样本数({total_needed_1})超过可用数量({available_1})")

        print(f"\n开始双组不重复采样:")
        print(f"  第一组 - 类别0: {group1_class_0} 条, 类别1: {group1_class_1} 条")
        print(f"  第二组 - 类别0: {group2_class_0} 条, 类别1: {group2_class_1} 条")
        print(f"  总需求 - 类别0: {total_needed_0} 条, 类别1: {total_needed_1} 条")

        # 设置随机种子
        np.random.seed(random_state)

        # 分离不同类别的数据
        class_0_data = self.df[self.df[self.target_column] == 0].reset_index(drop=True)
        class_1_data = self.df[self.df[self.target_column] == 1].reset_index(drop=True)

        # 随机打乱数据顺序
        class_0_shuffled = class_0_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        class_1_shuffled = class_1_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # 为两组分配不重复的样本
        # 类别0的分配
        group1_class_0_data = class_0_shuffled.iloc[:group1_class_0]
        group2_class_0_data = class_0_shuffled.iloc[group1_class_0:group1_class_0 + group2_class_0]

        # 类别1的分配
        group1_class_1_data = class_1_shuffled.iloc[:group1_class_1]
        group2_class_1_data = class_1_shuffled.iloc[group1_class_1:group1_class_1 + group2_class_1]

        # 合并每组的数据
        group1_data = pd.concat([group1_class_0_data, group1_class_1_data], ignore_index=True)
        group2_data = pd.concat([group2_class_0_data, group2_class_1_data], ignore_index=True)

        # 随机打乱每组内部的顺序
        group1_data = group1_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        group2_data = group2_data.sample(frac=1, random_state=random_state + 1).reset_index(drop=True)

        print(f"\n采样完成!")
        print(f"第一组数据形状: {group1_data.shape}")
        print(f"第二组数据形状: {group2_data.shape}")

        # 验证采样结果
        group1_counts = group1_data[self.target_column].value_counts().sort_index()
        group2_counts = group2_data[self.target_column].value_counts().sort_index()

        print(f"\n第一组数据分布:")
        for class_label, count in group1_counts.items():
            print(f"  类别 {class_label}: {count} 条记录 ({count / len(group1_data) * 100:.1f}%)")

        print(f"\n第二组数据分布:")
        for class_label, count in group2_counts.items():
            print(f"  类别 {class_label}: {count} 条记录 ({count / len(group2_data) * 100:.1f}%)")

        # 验证两组数据没有重复
        # 这里我们使用所有列来检查是否有完全相同的行
        merged_check = pd.concat([group1_data, group2_data])
        duplicates = merged_check.duplicated().sum()
        print(f"\n重复验证: 两组间重复样本数量: {duplicates}")

        # 保存结果
        if output_file1 is None:
            output_file1 = f"group1_data_{group1_class_0}_{group1_class_1}.csv"
        if output_file2 is None:
            output_file2 = f"group2_data_{group2_class_0}_{group2_class_1}.csv"

        group1_data.to_csv(output_file1, index=False)
        group2_data.to_csv(output_file2, index=False)

        print(f"\n第一组数据已保存到: {output_file1}")
        print(f"第二组数据已保存到: {output_file2}")

        return group1_data, group2_data

    def balanced_sample(self, n_class_0, n_class_1, random_state=42, output_file=None):
        """
        平衡采样

        参数:
        n_class_0: int, 类别0的样本数量
        n_class_1: int, 类别1的样本数量
        random_state: int, 随机种子
        output_file: str, 输出文件名，如果为None则自动生成

        返回:
        pd.DataFrame, 采样后的数据
        """
        if self.df is None:
            raise ValueError("请先使用 load_data() 加载数据")

        # 检查请求的样本数量是否超过可用数量
        available_0 = self.class_counts.get(0, 0)
        available_1 = self.class_counts.get(1, 0)

        if n_class_0 > available_0:
            print(f"警告: 请求的类别0样本数量({n_class_0})超过可用数量({available_0})")
            print(f"将使用所有可用的类别0样本({available_0})")
            n_class_0 = available_0

        if n_class_1 > available_1:
            print(f"警告: 请求的类别1样本数量({n_class_1})超过可用数量({available_1})")
            print(f"将使用所有可用的类别1样本({available_1})")
            n_class_1 = available_1

        # 分别采样各类别
        print(f"\n开始采样:")
        print(f"  类别0: {n_class_0} 条样本")
        print(f"  类别1: {n_class_1} 条样本")

        # 设置随机种子
        np.random.seed(random_state)

        # 分离不同类别的数据
        class_0_data = self.df[self.df[self.target_column] == 0]
        class_1_data = self.df[self.df[self.target_column] == 1]

        # 随机采样
        sampled_0 = resample(class_0_data, n_samples=n_class_0,
                             random_state=random_state, replace=False)
        sampled_1 = resample(class_1_data, n_samples=n_class_1,
                             random_state=random_state, replace=False)

        # 合并采样结果
        balanced_data = pd.concat([sampled_0, sampled_1], ignore_index=True)

        # 随机打乱顺序
        balanced_data = balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

        print(f"\n采样完成!")
        print(f"最终数据形状: {balanced_data.shape}")

        # 验证采样结果
        final_counts = balanced_data[self.target_column].value_counts().sort_index()
        print(f"最终数据分布:")
        for class_label, count in final_counts.items():
            print(f"  类别 {class_label}: {count} 条记录 ({count / len(balanced_data) * 100:.1f}%)")

        # 保存结果
        if output_file is None:
            output_file = f"balanced_data_{n_class_0}_{n_class_1}.csv"

        balanced_data.to_csv(output_file, index=False)
        print(f"\n平衡采样结果已保存到: {output_file}")

        return balanced_data

    def stratified_sample(self, total_samples, class_0_ratio=0.5, random_state=42, output_file=None):
        """
        分层采样，按比例采样

        参数:
        total_samples: int, 总样本数
        class_0_ratio: float, 类别0的比例，默认0.5（即50%）
        random_state: int, 随机种子
        output_file: str, 输出文件名

        返回:
        pd.DataFrame, 采样后的数据
        """
        n_class_0 = int(total_samples * class_0_ratio)
        n_class_1 = total_samples - n_class_0

        print(f"分层采样参数:")
        print(f"  总样本数: {total_samples}")
        print(f"  类别0比例: {class_0_ratio:.2%}")
        print(f"  类别0样本数: {n_class_0}")
        print(f"  类别1样本数: {n_class_1}")

        return self.balanced_sample(n_class_0, n_class_1, random_state, output_file)

    def undersample_majority(self, random_state=42, output_file=None):
        """
        对多数类进行欠采样，使其与少数类数量相等

        参数:
        random_state: int, 随机种子
        output_file: str, 输出文件名

        返回:
        pd.DataFrame, 采样后的数据
        """
        if self.df is None:
            raise ValueError("请先使用 load_data() 加载数据")

        # 找出少数类的数量
        min_class_count = self.class_counts.min()

        print(f"欠采样多数类:")
        print(f"  目标样本数（每类）: {min_class_count}")

        return self.balanced_sample(min_class_count, min_class_count, random_state, output_file)


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='数据平衡采样工具')
    parser.add_argument('csv_file', help='输入CSV文件路径')
    parser.add_argument('--target_column', default='HeightLabel', help='目标列名')

    # 双组采样参数
    parser.add_argument('--dual_group', action='store_true', help='启用双组不重复采样')
    parser.add_argument('--group1_class_0', type=int, help='第一组类别0的样本数量')
    parser.add_argument('--group1_class_1', type=int, help='第一组类别1的样本数量')
    parser.add_argument('--group2_class_0', type=int, help='第二组类别0的样本数量')
    parser.add_argument('--group2_class_1', type=int, help='第二组类别1的样本数量')
    parser.add_argument('--output1', help='第一组输出文件名')
    parser.add_argument('--output2', help='第二组输出文件名')

    # 原有的单组采样参数
    parser.add_argument('--n_class_0', type=int, help='类别0的样本数量')
    parser.add_argument('--n_class_1', type=int, help='类别1的样本数量')
    parser.add_argument('--total_samples', type=int, help='总样本数（用于分层采样）')
    parser.add_argument('--class_0_ratio', type=float, default=0.5, help='类别0的比例（用于分层采样）')
    parser.add_argument('--undersample', action='store_true', help='对多数类进行欠采样')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--output', help='输出文件名')

    args = parser.parse_args()

    # 创建采样器
    sampler = DataBalanceSampler(args.csv_file, args.target_column)

    # 加载数据
    sampler.load_data()

    # 根据参数选择采样方法
    if args.dual_group:
        # 双组不重复采样
        if not all([args.group1_class_0, args.group1_class_1, args.group2_class_0, args.group2_class_1]):
            print("错误: 双组采样需要指定所有组的样本数量")
            print("请使用: --group1_class_0, --group1_class_1, --group2_class_0, --group2_class_1")
            return

        result = sampler.dual_group_sample(
            args.group1_class_0, args.group1_class_1,
            args.group2_class_0, args.group2_class_1,
            args.random_state, args.output1, args.output2
        )
    elif args.undersample:
        # 欠采样多数类
        result = sampler.undersample_majority(args.random_state, args.output)
    elif args.total_samples:
        # 分层采样
        result = sampler.stratified_sample(args.total_samples, args.class_0_ratio,
                                           args.random_state, args.output)
    elif args.n_class_0 and args.n_class_1:
        # 平衡采样
        result = sampler.balanced_sample(args.n_class_0, args.n_class_1,
                                         args.random_state, args.output)
    else:
        print("错误: 请指定采样参数")
        print("选项1: --n_class_0 和 --n_class_1")
        print("选项2: --total_samples 和 --class_0_ratio")
        print("选项3: --undersample")
        print("选项4: --dual_group 和相关参数")
        return

    print(f"\n采样完成！")


# 使用示例
if __name__ == "__main__":
    # 如果直接运行脚本，使用示例参数
    if len(os.sys.argv) == 1:
        print("数据平衡采样工具使用示例:\n")

        # 示例1: 直接使用类
        print("=== 示例1: 直接使用采样器类 ===")
        sampler = DataBalanceSampler('../../merged_train_data.csv')

        try:
            # 加载数据
            sampler.load_data()

            # 新功能示例：双组不重复采样
            print("\n=== 双组不重复采样示例 ===")
            group1_data, group2_data = sampler.dual_group_sample(
                group1_class_0=16500, group1_class_1=38580,  # 第一组：4000个类别0，4000个类别1
                group2_class_0=16000, group2_class_1=30000,  # 第二组：3000个类别0，3000个类别1
                output_file1='train_group1.csv',
                output_file2='train_group2.csv'
            )

            # # 原有功能示例：平衡采样
            # print("\n=== 平衡采样示例 ===")
            # balanced_data = sampler.balanced_sample(n_class_0=8000, n_class_1=8000,
            #                                         output_file='balanced_8k_8k.csv')
            #
            # # 分层采样示例
            # print("\n=== 分层采样示例 ===")
            # stratified_data = sampler.stratified_sample(total_samples=20000,
            #                                             class_0_ratio=0.3,
            #                                             output_file='stratified_20k.csv')
            #
            # # 欠采样示例
            # print("\n=== 欠采样示例 ===")
            # undersampled_data = sampler.undersample_majority(output_file='undersampled.csv')

        except FileNotFoundError:
            print("示例文件不存在，请参考命令行用法")

        print("\n=== 命令行用法示例 ===")
        print("# 双组不重复采样")
        print(
            "python data_sampler.py Train_OD3.csv --dual_group --group1_class_0 4000 --group1_class_1 4000 --group2_class_0 3000 --group2_class_1 3000")
        # print("\n# 单组平衡采样")
        # print("python data_sampler.py Train_OD3.csv --n_class_0 8000 --n_class_1 8000")
        # print("\n# 分层采样")
        # print("python data_sampler.py Train_OD3.csv --total_samples 20000 --class_0_ratio 0.3")
        # print("\n# 欠采样")
        # print("python data_sampler.py Train_OD3.csv --undersample")

    else:
        # 使用命令行参数
        main()