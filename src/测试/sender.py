import socket
import time
import struct

# 配置
TARGET_IP = "127.0.0.1"  # 本地回环地址
TARGET_PORT = 8888
LOCAL_PORT = 9999


def send_test_packet(data, description=""):
    """发送单个测试数据包"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', LOCAL_PORT))

    try:
        sock.sendto(data, (TARGET_IP, TARGET_PORT))
        print(f"✓ 发送 {description}: {len(data)} 字节")
    except Exception as e:
        print(f"❌ 发送失败: {e}")
    finally:
        sock.close()


def main():
    print("=== UDP数据包发送器 ===")

    # 测试1: 简单文本
    send_test_packet(b"Hello UDP!", "文本消息")
    time.sleep(1)

    # 测试2: 模拟原程序格式（长度+命令+数据）
    message = b"Test message data"
    length = len(message) + 1  # +1 for command byte
    packet = struct.pack('<I', length) + b'\x01' + message
    send_test_packet(packet, "结构化数据包")
    time.sleep(1)

    # 测试3: 二进制数据
    binary_data = bytes(range(0, 256, 10))
    send_test_packet(binary_data, "二进制数据")
    time.sleep(1)

    # 测试4: 大数据包
    large_data = b"X" * 1000
    send_test_packet(large_data, "大数据包(1KB)")

    print("所有测试数据包已发送完成!")


if __name__ == "__main__":
    main()