import re
import pandas as pd
from collections import defaultdict, OrderedDict
import numpy as np


def parse_log_line(line):
    """解析单行日志数据"""
    # 使用正则表达式提取各个字段
    pattern = r'zhao_od_ID:(\d+), ObjName:([^,]+), HeightLabel:(\d+), OdDir:(\d+), P1\(([^)]+)\), P2\(([^)]+)\), HeightProb:(\d+), HeightStatus:(\d+), Dist:(\d+), PosDeDis1:(\d+), l_TrainResult:(\d+), probability:(\d+)'
    match = re.match(pattern, line.strip())

    if match:
        return {
            'zhao_od_ID': int(match.group(1)),
            'ObjName': match.group(2),
            'HeightLabel': int(match.group(3)),
            'OdDir': int(match.group(4)),
            'P1': match.group(5),
            'P2': match.group(6),
            'HeightProb': int(match.group(7)),
            'HeightStatus': int(match.group(8)),
            'Dist': int(match.group(9)),
            'PosDeDis1': int(match.group(10)),
            'l_TrainResult': int(match.group(11)),
            'probability': int(match.group(12))
        }
    return None


def identify_test_sessions(data_list):
    """识别测试次数，根据Dist变化判断"""
    if len(data_list) <= 1:
        return [data_list]

    sessions = []
    current_session = [data_list[0]]

    for i in range(1, len(data_list)):
        prev_dist = data_list[i - 1]['Dist']
        curr_dist = data_list[i]['Dist']

        # 如果距离突然增大很多（超过前一个距离的1.5倍），认为是新的一次测试
        if curr_dist > prev_dist * 1.5 and curr_dist > prev_dist + 500:
            sessions.append(current_session)
            current_session = [data_list[i]]
        else:
            current_session.append(data_list[i])

    if current_session:
        sessions.append(current_session)

    return sessions


def analyze_height_status_changes(session_data):
    """分析单次测试中HeightStatus的变化"""
    # 按距离排序（从大到小，接近过程）
    session_data.sort(key=lambda x: x['Dist'], reverse=True)

    changes = []
    current_status = session_data[0]['HeightStatus']
    change_start_idx = 0

    for i, data in enumerate(session_data):
        if data['HeightStatus'] != current_status:
            # 记录状态变化
            changes.append({
                'start_idx': change_start_idx,
                'end_idx': i - 1,
                'status': current_status,
                'start_dist': session_data[change_start_idx]['Dist'],
                'end_dist': session_data[i - 1]['Dist']
            })
            current_status = data['HeightStatus']
            change_start_idx = i

    # 添加最后一段
    if change_start_idx < len(session_data):
        changes.append({
            'start_idx': change_start_idx,
            'end_idx': len(session_data) - 1,
            'status': current_status,
            'start_dist': session_data[change_start_idx]['Dist'],
            'end_dist': session_data[-1]['Dist']
        })

    return changes


def format_distance_ranges(changes, height_label):
    """格式化距离区间信息"""
    if not changes:
        return "无数据"

    # 找出错误报告的区间（HeightStatus与HeightLabel不一致）
    wrong_ranges = []
    correct_ranges = []
    first_wrong_dist = None

    for change in changes:
        status_desc = "报高" if change['status'] == 2 else "报低"
        range_desc = f"Dist:{change['start_dist']}-{change['end_dist']}"

        # 判断是否与真实标签一致
        is_correct = (height_label == 0 and change['status'] == 1) or (height_label == 1 and change['status'] == 2)

        if is_correct:
            correct_ranges.append(f"{range_desc}({status_desc})")
        else:
            wrong_ranges.append(f"{range_desc}({status_desc})")
            if first_wrong_dist is None:
                first_wrong_dist = change['start_dist']

    # 构建描述
    descriptions = []
    if first_wrong_dist is not None:
        descriptions.append(f"首次错误报告距离:{first_wrong_dist}")

    if wrong_ranges:
        descriptions.append(f"错误区间:{'; '.join(wrong_ranges)}")

    if correct_ranges:
        descriptions.append(f"正确区间:{'; '.join(correct_ranges)}")

    # 检查是否在某个距离后全程正确
    if len(changes) > 1:
        last_change = changes[-1]
        last_is_correct = (height_label == 0 and last_change['status'] == 1) or (
                    height_label == 1 and last_change['status'] == 2)
        if last_is_correct and last_change['end_dist'] < last_change['start_dist'] * 0.7:
            descriptions.append(f"Dist<{last_change['start_dist']}后全程{'报低' if height_label == 0 else '报高'}")

    return "; ".join(descriptions) if descriptions else "全程正确"


def get_status_analysis(height_label, has_error, changes):
    """获取状态分析描述"""
    if not has_error:
        return "全程正确"

    # 检查具体的错误类型
    for change in changes:
        is_error = not ((height_label == 0 and change['status'] == 1) or
                        (height_label == 1 and change['status'] == 2))
        if is_error:
            if height_label == 0 and change['status'] == 2:  # 低障碍物但报高
                return "存在报高"
            elif height_label == 1 and change['status'] == 1:  # 高障碍物但报低
                return "存在报低"

    return "存在错误"


def analyze_obstacle_data(log_file_path, output_csv_path):
    """主要分析函数"""
    # 读取日志文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析所有数据
    all_data = []
    for line in lines:
        parsed = parse_log_line(line)
        if parsed:
            all_data.append(parsed)

    # 按障碍物分组 (ObjName + OdDir + HeightLabel)
    obstacle_groups = defaultdict(list)
    for data in all_data:
        key = (data['ObjName'], data['OdDir'], data['HeightLabel'])
        obstacle_groups[key].append(data)

    # 详细结果和简洁结果
    detailed_results = []
    summary_results = []

    for (obj_name, od_dir, height_label), data_list in obstacle_groups.items():
        # 转换OdDir为可读格式
        od_dir_str = "front" if od_dir == 1 else "rear"
        height_label_str = "低障碍物" if height_label == 0 else "高障碍物"

        # 按zhao_od_ID分组
        id_groups = defaultdict(list)
        for data in data_list:
            id_groups[data['zhao_od_ID']].append(data)

        # 用于简洁版统计
        obstacle_has_any_error = False
        total_sessions_all_ids = 0
        total_points_all_ids = 0

        for zhao_od_id, id_data in id_groups.items():
            # 识别测试次数
            sessions = identify_test_sessions(id_data)
            total_sessions_all_ids += len(sessions)

            # 检查所有次数是否都正确
            all_sessions_correct = True
            incorrect_sessions = []

            for session_idx, session in enumerate(sessions, 1):
                session_correct = all(
                    (height_label == 0 and data['HeightStatus'] == 1) or
                    (height_label == 1 and data['HeightStatus'] == 2)
                    for data in session
                )

                if not session_correct:
                    all_sessions_correct = False
                    incorrect_sessions.append(session_idx)
                    obstacle_has_any_error = True

            # 统计总数据点
            total_points_all_ids += sum(len(session) for session in sessions)

            # 如果所有次数都正确，只输出一条记录
            if all_sessions_correct:
                total_points = sum(len(session) for session in sessions)
                min_dist = min(data['Dist'] for session in sessions for data in session)
                max_dist = max(data['Dist'] for session in sessions for data in session)

                detailed_results.append({
                    '障碍物名称': obj_name,
                    '位置': od_dir_str,
                    '真实标签': height_label_str,
                    '障碍物ID': zhao_od_id,
                    '总测试次数': len(sessions),
                    '测试次数说明': f"第{','.join(map(str, range(1, len(sessions) + 1)))}次" if len(
                        sessions) > 1 else "第1次",
                    '数据点总数': total_points,
                    '距离范围(mm)': f"{max_dist}~{min_dist}",
                    '检测状态': "全程正确",
                    '首次错误距离(mm)': "无",
                    '错误详情': "无",
                    '备注': f"共{len(sessions)}次测试，全程HeightStatus与HeightLabel一致"
                })

            else:
                # 对于错误的次数，分别输出
                for session_idx, session in enumerate(sessions, 1):
                    if session_idx in incorrect_sessions:
                        changes = analyze_height_status_changes(session)
                        error_desc = format_distance_ranges(changes, height_label)
                        status_analysis = get_status_analysis(height_label, True, changes)

                        first_error_dist = None
                        for change in changes:
                            is_error = not ((height_label == 0 and change['status'] == 1) or
                                            (height_label == 1 and change['status'] == 2))
                            if is_error:
                                first_error_dist = change['start_dist']
                                break

                        detailed_results.append({
                            '障碍物名称': obj_name,
                            '位置': od_dir_str,
                            '真实标签': height_label_str,
                            '障碍物ID': zhao_od_id,
                            '总测试次数': len(sessions),
                            '测试次数说明': f"第{session_idx}次",
                            '数据点总数': len(session),
                            '距离范围(mm)': f"{max(d['Dist'] for d in session)}~{min(d['Dist'] for d in session)}",
                            '检测状态': status_analysis,
                            '首次错误距离(mm)': str(first_error_dist) if first_error_dist else "无",
                            '错误详情': error_desc,
                            '备注': f"第{session_idx}次测试中HeightStatus与HeightLabel不一致"
                        })

                # 对于正确的次数，合并输出一条记录
                correct_sessions = [i for i in range(1, len(sessions) + 1) if i not in incorrect_sessions]
                if correct_sessions:
                    correct_data = [sessions[i - 1] for i in correct_sessions]
                    total_points = sum(len(session) for session in correct_data)
                    min_dist = min(data['Dist'] for session in correct_data for data in session)
                    max_dist = max(data['Dist'] for session in correct_data for data in session)

                    detailed_results.append({
                        '障碍物名称': obj_name,
                        '位置': od_dir_str,
                        '真实标签': height_label_str,
                        '障碍物ID': zhao_od_id,
                        '总测试次数': len(sessions),
                        '测试次数说明': f"第{','.join(map(str, correct_sessions))}次",
                        '数据点总数': total_points,
                        '距离范围(mm)': f"{max_dist}~{min_dist}",
                        '检测状态': "全程正确",
                        '首次错误距离(mm)': "无",
                        '错误详情': "无",
                        '备注': f"第{','.join(map(str, correct_sessions))}次测试全程正确"
                    })

        # 生成简洁版结果
        if not obstacle_has_any_error:
            # 全程无错误，不区分ID
            all_distances = [data['Dist'] for data in data_list]
            summary_results.append({
                '障碍物名称': obj_name,
                '位置': od_dir_str,
                '真实标签': height_label_str,
                '总测试次数': total_sessions_all_ids,
                '涉及ID数量': len(id_groups),
                '数据点总数': total_points_all_ids,
                '距离范围(mm)': f"{max(all_distances)}~{min(all_distances)}",
                '检测结果': "✓ 全程正确",
                '备注': f"所有ID({','.join(map(str, sorted(id_groups.keys())))})全程检测正确"
            })
        else:
            # 存在错误，需要查看详细表
            error_type = "存在报高" if height_label == 0 else "存在报低"
            all_distances = [data['Dist'] for data in data_list]
            summary_results.append({
                '障碍物名称': obj_name,
                '位置': od_dir_str,
                '真实标签': height_label_str,
                '总测试次数': total_sessions_all_ids,
                '涉及ID数量': len(id_groups),
                '数据点总数': total_points_all_ids,
                '距离范围(mm)': f"{max(all_distances)}~{min(all_distances)}",
                '检测结果': f"✗ {error_type}",
                '备注': "存在检测错误，详情请查看详细分析表"
            })

    # 创建输出文件名
    base_name = output_csv_path.replace('.csv', '')
    detailed_csv = f"{base_name}_详细分析.csv"
    summary_csv = f"{base_name}_简洁汇总.csv"

    # 转换为DataFrame并保存
    detailed_df = pd.DataFrame(detailed_results)
    summary_df = pd.DataFrame(summary_results)

    # 保存详细分析表
    detailed_df.to_csv(detailed_csv, index=False, encoding='utf-8-sig')

    # 保存简洁汇总表
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

    # 美化输出
    print("=" * 80)
    print("🎯 障碍物检测日志分析完成!")
    print("=" * 80)
    print(f"📁 详细分析表: {detailed_csv}")
    print(f"📊 简洁汇总表: {summary_csv}")
    print()

    print("📈 统计概览:")
    print("-" * 50)
    print(f"🔍 分析障碍物类型总数: {len(obstacle_groups)}")
    print(f"📝 详细记录条数: {len(detailed_results)}")
    print(f"📋 汇总记录条数: {len(summary_results)}")
    print(f"✅ 全程正确的障碍物: {len(summary_df[summary_df['检测结果'].str.contains('全程正确')])}")
    print(f"❌ 存在错误的障碍物: {len(summary_df[summary_df['检测结果'].str.contains('存在')])}")
    print()

    print("📋 简洁汇总表预览:")
    print("-" * 50)
    if len(summary_df) > 0:
        # 设置pandas显示选项以更好地显示中文
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        print(summary_df.head(10).to_string(index=False))
        if len(summary_df) > 10:
            print(f"\n... 还有 {len(summary_df) - 10} 行数据，详见CSV文件")
    else:
        print("无数据")

    print("\n" + "=" * 80)

    return detailed_df, summary_df


# 使用示例
if __name__ == "__main__":
    # 修改这里的路径为你的实际文件路径
    log_file_path = r"D:\PythonProject\data\log_files\1.log"  # 输入的日志文件路径
    output_csv_path = r"D:\PythonProject\data\csv_files\1.csv"   # 输出的CSV文件路径

    try:
        detailed_df, summary_df = analyze_obstacle_data(log_file_path, output_csv_path)

        print("\n🔍 详细分析表字段说明:")
        print("-" * 50)
        field_descriptions = {
            '障碍物名称': '障碍物类型名称',
            '位置': 'front(车前) / rear(车后)',
            '真实标签': '实际的高低标签',
            '障碍物ID': 'zhao_od_ID编号',
            '总测试次数': '该ID的测试次数',
            '测试次数说明': '当前记录对应的测试次数',
            '数据点总数': '数据记录条数',
            '距离范围(mm)': '测试距离范围(最远~最近)',
            '检测状态': '全程正确/存在报高/存在报低',
            '首次错误距离(mm)': '第一次出现错误时的距离',
            '错误详情': '错误的具体距离区间信息',
            '备注': '额外说明信息'
        }

        for field, desc in field_descriptions.items():
            print(f"  • {field}: {desc}")

        print("\n📊 简洁汇总表说明:")
        print("-" * 50)
        print("  • 全程无错误的障碍物：不区分不同ID，合并显示")
        print("  • 存在错误的障碍物：标注错误类型，详情见详细分析表")
        print("  • ✓ 表示检测正确，✗ 表示存在错误")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到日志文件 '{log_file_path}'")
        print("请确保文件路径正确，或将日志内容保存为 'test.log' 文件")
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback

        traceback.print_exc()