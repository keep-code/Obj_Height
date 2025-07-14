import pandas as pd
import os
import re
import chardet


class CSVMerger:
    def __init__(self):
        self.supported_encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'utf-8-sig']

    def detect_encoding(self, file_path):
        """自动检测文件编码"""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding']

    def read_csv_with_encoding(self, file_path):
        """尝试不同编码读取CSV文件"""
        # 首先尝试自动检测编码
        detected_encoding = self.detect_encoding(file_path)
        if detected_encoding:
            try:
                with open(file_path, 'r', encoding=detected_encoding) as f:
                    content = f.read()
                return content, detected_encoding
            except UnicodeDecodeError:
                pass

        # 如果自动检测失败，尝试预设的编码
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
            with open(temp_file, 'w', encoding='utf-8', newline='') as f:
                f.write(table_content)

            try:
                df = pd.read_csv(temp_file, encoding='utf-8')
                tables.append(df)
            finally:
                # 删除临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        return tables, header_pattern

    def merge_csv_files(self, source_file, target_file, output_file=None, output_encoding='utf-8-sig'):
        """
        将源文件的数据合并到目标文件

        Args:
            source_file: 源CSV文件路径
            target_file: 目标CSV文件路径
            output_file: 输出文件路径（如果为None，则覆盖目标文件）
            output_encoding: 输出文件编码，默认utf-8-sig（带BOM的UTF-8，Excel兼容）
        """
        # 读取源文件和目标文件
        print(f"正在读取源文件: {source_file}")
        source_content, source_encoding = self.read_csv_with_encoding(source_file)
        print(f"源文件编码: {source_encoding}")

        print(f"正在读取目标文件: {target_file}")
        target_content, target_encoding = self.read_csv_with_encoding(target_file)
        print(f"目标文件编码: {target_encoding}")

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

        # 使用指定编码保存，并确保正确处理中文
        merged_df.to_csv(output_file, index=False, encoding=output_encoding)

        print(f"成功合并文件!")
        print(f"源文件 {source_file}: {sum(len(df) for df in source_tables)} 行数据")
        print(f"目标文件 {target_file}: {sum(len(df) for df in target_tables)} 行数据")
        print(f"合并后文件 {output_file}: {len(merged_df)} 行数据")
        print(f"输出文件编码: {output_encoding}")

        return merged_df

    def merge_subtables_in_file(self, file_path, output_file=None, output_encoding='utf-8-sig'):
        """
        合并同一文件中的多个子表

        Args:
            file_path: CSV文件路径
            output_file: 输出文件路径（如果为None，则覆盖原文件）
            output_encoding: 输出文件编码
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

        merged_df.to_csv(output_file, index=False, encoding=output_encoding)

        print(f"成功合并文件中的子表!")
        print(f"原文件有 {len(tables)} 个子表")
        print(f"各子表行数: {[len(df) for df in tables]}")
        print(f"合并后文件 {output_file}: {len(merged_df)} 行数据")
        print(f"输出文件编码: {output_encoding}")

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

    def fix_encoding_for_existing_file(self, input_file, output_file=None, target_encoding='utf-8-sig'):
        """
        修复已有文件的编码问题

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径（如果为None，则覆盖原文件）
            target_encoding: 目标编码
        """
        print(f"正在修复文件编码: {input_file}")

        # 读取文件
        content, current_encoding = self.read_csv_with_encoding(input_file)
        print(f"当前文件编码: {current_encoding}")

        # 读取为DataFrame
        df = pd.read_csv(input_file, encoding=current_encoding)

        # 保存为指定编码
        if output_file is None:
            output_file = input_file

        df.to_csv(output_file, index=False, encoding=target_encoding)
        print(f"文件编码已修复为: {target_encoding}")
        print(f"输出文件: {output_file}")

        return df


# 使用示例
if __name__ == "__main__":
    merger = CSVMerger()

    # 修复现有的乱码文件
    print("=== 修复现有乱码文件 ===")
    try:
        merger.fix_encoding_for_existing_file(
            input_file="../../merged_train_data.csv",
            output_file="../../merged_train_data_fixed.csv",
            target_encoding='utf-8-sig'
        )
    except Exception as e:
        print(f"修复文件编码时出错: {e}")

    # 重新合并文件（使用正确的编码）
    print("\n=== 重新合并两个CSV文件 ===")
    try:
        merged_df = merger.merge_csv_files(
            source_file="../../Train_OD_low.csv",
            target_file="../../Train_OD_4.csv",
            output_file="../../merged_train_data_new.csv",
            output_encoding='utf-8-sig'  # 使用带BOM的UTF-8，Excel兼容
        )
    except Exception as e:
        print(f"合并文件时出错: {e}")

    # 验证合并结果
    print("\n=== 验证合并结果 ===")
    try:
        df = pd.read_csv("../../merged_train_data_new.csv", encoding='utf-8-sig')
        print(f"验证成功！合并后文件行数: {len(df)}")
        print(f"列名: {list(df.columns)}")

        # 显示前几行数据以确认中文显示正常
        print("\n前5行数据:")
        print(df.head())

    except Exception as e:
        print(f"验证文件时出错: {e}")