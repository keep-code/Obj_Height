import pandas as pd
import os
import re


class CSVMerger:
    def __init__(self):
        self.supported_encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']

    def read_csv_with_encoding(self, file_path):
        """尝试不同编码读取CSV文件"""
        for encoding in self.supported_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content, encoding
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"无法读取文件 {file_path}，尝试了所有支持的编码")

    def find_header_positions(self, content, header_pattern=None):
        """查找文件中所有表头的位置"""
        lines = content.split('\n')
        if header_pattern is None:
            # 使用第一行作为表头模式
            header_pattern = lines[0].strip()

        header_positions = []
        for i, line in enumerate(lines):
            if line.strip() == header_pattern:
                header_positions.append(i)

        return header_positions, header_pattern

    def split_multi_table_csv(self, file_path):
        """将包含多个子表的CSV文件分割成多个DataFrame"""
        content, encoding = self.read_csv_with_encoding(file_path)
        lines = content.split('\n')

        # 查找所有表头位置
        header_positions, header_pattern = self.find_header_positions(content)

        if len(header_positions) <= 1:
            # 只有一个表头，直接读取
            df = pd.read_csv(file_path, encoding=encoding)
            return [df], header_pattern

        # 多个表头，分割成多个表
        tables = []
        for i, start_pos in enumerate(header_positions):
            if i == len(header_positions) - 1:
                # 最后一个表，读取到文件末尾
                end_pos = len(lines)
            else:
                # 读取到下一个表头之前
                end_pos = header_positions[i + 1]

            # 提取当前表的数据
            table_lines = lines[start_pos:end_pos]
            table_content = '\n'.join(table_lines)

            # 创建临时文件来读取DataFrame
            temp_file = f"temp_table_{i}.csv"
            with open(temp_file, 'w', encoding=encoding) as f:
                f.write(table_content)

            try:
                df = pd.read_csv(temp_file, encoding=encoding)
                tables.append(df)
            finally:
                # 删除临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        return tables, header_pattern

    def merge_csv_files(self, source_file, target_file, output_file=None):
        """
        将源文件的数据合并到目标文件

        Args:
            source_file: 源CSV文件路径
            target_file: 目标CSV文件路径
            output_file: 输出文件路径（如果为None，则覆盖目标文件）
        """
        # 读取源文件和目标文件
        source_content, source_encoding = self.read_csv_with_encoding(source_file)
        target_content, target_encoding = self.read_csv_with_encoding(target_file)

        # 检查是否有多个子表
        source_tables, source_header = self.split_multi_table_csv(source_file)
        target_tables, target_header = self.split_multi_table_csv(target_file)

        # 检查表头是否一致
        if source_header != target_header:
            raise ValueError(f"表头不一致!\n源文件表头: {source_header}\n目标文件表头: {target_header}")

        # 合并所有表
        all_tables = target_tables + source_tables
        merged_df = pd.concat(all_tables, ignore_index=True)

        # 保存结果
        if output_file is None:
            output_file = target_file

        merged_df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"成功合并文件!")
        print(f"源文件 {source_file}: {sum(len(df) for df in source_tables)} 行数据")
        print(f"目标文件 {target_file}: {sum(len(df) for df in target_tables)} 行数据")
        print(f"合并后文件 {output_file}: {len(merged_df)} 行数据")

        return merged_df

    def merge_subtables_in_file(self, file_path, output_file=None):
        """
        合并同一文件中的多个子表

        Args:
            file_path: CSV文件路径
            output_file: 输出文件路径（如果为None，则覆盖原文件）
        """
        # 分割多个子表
        tables, header_pattern = self.split_multi_table_csv(file_path)

        if len(tables) <= 1:
            print(f"文件 {file_path} 只有一个表，无需合并")
            return tables[0] if tables else None

        # 合并所有子表
        merged_df = pd.concat(tables, ignore_index=True)

        # 保存结果
        if output_file is None:
            output_file = file_path

        merged_df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"成功合并文件中的子表!")
        print(f"原文件有 {len(tables)} 个子表")
        print(f"各子表行数: {[len(df) for df in tables]}")
        print(f"合并后文件 {output_file}: {len(merged_df)} 行数据")

        return merged_df

    def analyze_csv_structure(self, file_path):
        """分析CSV文件结构"""
        content, encoding = self.read_csv_with_encoding(file_path)
        lines = content.split('\n')

        header_positions, header_pattern = self.find_header_positions(content)

        print(f"\n文件分析: {file_path}")
        print(f"编码: {encoding}")
        print(f"总行数: {len(lines)}")
        print(f"表头数量: {len(header_positions)}")
        print(f"表头位置: {header_positions}")
        print(f"表头内容: {header_pattern}")

        if len(header_positions) > 1:
            print("子表信息:")
            for i, start_pos in enumerate(header_positions):
                if i == len(header_positions) - 1:
                    end_pos = len(lines)
                else:
                    end_pos = header_positions[i + 1]

                data_rows = end_pos - start_pos - 1  # 减去表头行
                print(f"  子表 {i + 1}: 第{start_pos + 1}行到第{end_pos}行，数据行数: {data_rows}")


# 使用示例
if __name__ == "__main__":
    merger = CSVMerger()

    # 示例1: 分析文件结构
    print("=== 分析文件结构 ===")
    merger.analyze_csv_structure("Train_OD_4.csv")
    merger.analyze_csv_structure("Train_OD_low.csv")

    # 示例2: 合并两个CSV文件
    print("\n=== 合并两个CSV文件 ===")
    try:
        merged_df = merger.merge_csv_files(
            source_file="../../Train_OD_low.csv",
            target_file="../../Train_OD_4.csv",
            output_file="../../merged_train_data.csv"
        )
    except Exception as e:
        print(f"合并文件时出错: {e}")

    # 示例3: 合并单个文件中的多个子表
    print("\n=== 合并Train_OD_4.csv中的子表 ===")
    try:
        merged_df = merger.merge_subtables_in_file(
            file_path="../../Train_OD_4.csv",
            output_file="Train_OD_4_merged.csv"
        )
    except Exception as e:
        print(f"合并子表时出错: {e}")

    # # 示例4: 自定义合并操作
    # print("\n=== 自定义合并操作示例 ===")
    # # 你可以根据需要修改这部分代码
    # # 例如：先合并Train_OD_4.csv的子表，然后再与Train_OD_low.csv合并
    # try:
    #     # 步骤1: 合并Train_OD_4.csv的子表
    #     merger.merge_subtables_in_file("Train_OD_4.csv", "Train_OD_4_single.csv")
    #
    #     # 步骤2: 将Train_OD_low.csv合并到处理后的Train_OD_4.csv
    #     final_df = merger.merge_csv_files(
    #         source_file="Train_OD_low.csv",
    #         target_file="Train_OD_4_single.csv",
    #         output_file="final_merged_data.csv"
    #     )
    #
    #     print(f"最终合并完成！文件保存为: final_merged_data.csv")
    #     print(f"最终数据行数: {len(final_df)}")
    #
    # except Exception as e:
    #     print(f"自定义合并时出错: {e}")