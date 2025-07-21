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

    # 分析结果
    results = []

    for (obj_name, od_dir, height_label), data_list in obstacle_groups.items():
        # 按zhao_od_ID分组
        id_groups = defaultdict(list)
        for data in data_list:
            id_groups[data['zhao_od_ID']].append(data)

        for zhao_od_id, id_data in id_groups.items():
            # 识别测试次数
            sessions = identify_test_sessions(id_data)

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

            # 如果所有次数都正确，只输出一条记录
            if all_sessions_correct:
                total_points = sum(len(session) for session in sessions)
                min_dist = min(data['Dist'] for session in sessions for data in session)
                max_dist = max(data['Dist'] for session in sessions for data in session)

                results.append({
                    'ObjName': obj_name,
                    'OdDir': od_dir,
                    'HeightLabel': height_label,
                    'zhao_od_ID': zhao_od_id,
                    'TotalSessions': len(sessions),
                    'SessionInfo': f"第{','.join(map(str, range(1, len(sessions) + 1)))}次" if len(
                        sessions) > 1 else "第1次",
                    'TotalDataPoints': total_points,
                    'DistanceRange': f"{max_dist}-{min_dist}",
                    'StatusAnalysis': "全程正确",
                    'ErrorDescription': "无",
                    'FirstErrorDist': "无",
                    'Notes': f"共{len(sessions)}次测试，全程HeightStatus与HeightLabel一致"
                })

            else:
                # 对于错误的次数，分别输出
                for session_idx, session in enumerate(sessions, 1):
                    if session_idx in incorrect_sessions:
                        changes = analyze_height_status_changes(session)
                        error_desc = format_distance_ranges(changes, height_label)

                        first_error_dist = None
                        for change in changes:
                            is_error = not ((height_label == 0 and change['status'] == 1) or
                                            (height_label == 1 and change['status'] == 2))
                            if is_error:
                                first_error_dist = change['start_dist']
                                break

                        results.append({
                            'ObjName': obj_name,
                            'OdDir': od_dir,
                            'HeightLabel': height_label,
                            'zhao_od_ID': zhao_od_id,
                            'TotalSessions': len(sessions),
                            'SessionInfo': f"第{session_idx}次",
                            'TotalDataPoints': len(session),
                            'DistanceRange': f"{max(d['Dist'] for d in session)}-{min(d['Dist'] for d in session)}",
                            'StatusAnalysis': "存在错误",
                            'ErrorDescription': error_desc,
                            'FirstErrorDist': str(first_error_dist) if first_error_dist else "无",
                            'Notes': f"第{session_idx}次测试中HeightStatus与HeightLabel不一致"
                        })

                # 对于正确的次数，合并输出一条记录
                correct_sessions = [i for i in range(1, len(sessions) + 1) if i not in incorrect_sessions]
                if correct_sessions:
                    correct_data = [sessions[i - 1] for i in correct_sessions]
                    total_points = sum(len(session) for session in correct_data)
                    min_dist = min(data['Dist'] for session in correct_data for data in session)
                    max_dist = max(data['Dist'] for session in correct_data for data in session)

                    results.append({
                        'ObjName': obj_name,
                        'OdDir': od_dir,
                        'HeightLabel': height_label,
                        'zhao_od_ID': zhao_od_id,
                        'TotalSessions': len(sessions),
                        'SessionInfo': f"第{','.join(map(str, correct_sessions))}次",
                        'TotalDataPoints': total_points,
                        'DistanceRange': f"{max_dist}-{min_dist}",
                        'StatusAnalysis': "全程正确",
                        'ErrorDescription': "无",
                        'FirstErrorDist': "无",
                        'Notes': f"第{','.join(map(str, correct_sessions))}次测试全程正确"
                    })

    # 转换为DataFrame并保存
    df = pd.DataFrame(results)

    # 重新排序列
    column_order = [
        'ObjName', 'OdDir', 'HeightLabel', 'zhao_od_ID', 'TotalSessions',
        'SessionInfo', 'TotalDataPoints', 'DistanceRange', 'StatusAnalysis',
        'ErrorDescription', 'FirstErrorDist', 'Notes'
    ]

    df = df[column_order]
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"分析完成！结果已保存到: {output_csv_path}")
    print(f"共分析了 {len(obstacle_groups)} 个不同障碍物")
    print(f"生成了 {len(results)} 条统计记录")

    return df


# 使用示例
if __name__ == "__main__":
    # 修改这里的路径为你的实际文件路径
    log_file_path = r"D:\PythonProject\data\log_files\1.log"  # 输入的日志文件路径
    output_csv_path = r"D:\PythonProject\data\csv_files\1.csv"  # 输出的CSV文件路径

    try:
        results_df = analyze_obstacle_data(log_file_path, output_csv_path)

        # 显示前几行结果作为预览
        print("\n前5行结果预览:")
        print(results_df.head().to_string(index=False))

        # 显示一些统计信息
        print(f"\n统计信息:")
        print(f"- 总记录数: {len(results_df)}")
        print(f"- 存在错误的记录数: {len(results_df[results_df['StatusAnalysis'] == '存在错误'])}")
        print(f"- 全程正确的记录数: {len(results_df[results_df['StatusAnalysis'] == '全程正确'])}")

    except FileNotFoundError:
        print(f"错误: 找不到日志文件 '{log_file_path}'")
        print("请确保文件路径正确，或将日志内容保存为 'test.log' 文件")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")