import socket
import struct
import time
import threading
from typing import List, Dict, Tuple
import subprocess
import sys


class MCUPortScanner:
    """MCU端口扫描器 - 多种方法确定MCU端口"""

    def __init__(self, mcu_ip: str = "198.18.36.1", local_ip: str = "198.18.36.100"):
        self.mcu_ip = mcu_ip
        self.local_ip = local_ip
        self.scan_results = {}

    def method1_tcp_port_scan(self, port_range: range = range(1, 10000)) -> List[int]:
        """方法1: TCP端口扫描 - 找出开放的端口"""
        print(f"\n=== 方法1: TCP端口扫描 {self.mcu_ip} ===")
        print("扫描范围: 1-9999 (这可能需要几分钟)")

        open_ports = []
        scan_count = 0

        for port in port_range:
            scan_count += 1
            if scan_count % 1000 == 0:
                print(f"已扫描 {scan_count} 个端口...")

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)  # 100ms超时
                result = sock.connect_ex((self.mcu_ip, port))

                if result == 0:
                    print(f"✓ 发现开放的TCP端口: {port}")
                    open_ports.append(port)

                sock.close()

            except Exception:
                continue

        print(f"TCP扫描完成，发现 {len(open_ports)} 个开放端口: {open_ports}")
        return open_ports

    def method2_udp_response_scan(self, port_list: List[int] = None) -> List[int]:
        """方法2: UDP响应扫描 - 发送数据包看哪个端口有响应"""
        if port_list is None:
            port_list = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995,
                         502, 1000, 1234, 2000, 3000, 5000, 5555, 5556,
                         6000, 7000, 8000, 8080, 8888, 9000, 9999]

        print(f"\n=== 方法2: UDP响应扫描 ===")
        print(f"测试端口: {port_list}")

        responsive_ports = []

        for port in port_list:
            print(f"测试UDP端口 {port}...", end=" ")

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(2.0)
                sock.bind((self.local_ip, 0))  # 自动分配本地端口

                # 发送多种测试数据包
                test_packets = [
                    b'\x02\xFF\x00',  # XCP CONNECT
                    b'\x01\xFD',  # XCP GET_STATUS
                    b'\x00',  # 空数据包
                    b'\x01\x00',  # 简单测试
                    b'test',  # 文本测试
                    b'\x00\x00\x00\x00',  # 4字节测试
                ]

                got_response = False
                for packet in test_packets:
                    try:
                        sock.sendto(packet, (self.mcu_ip, port))
                        data, addr = sock.recvfrom(1024)

                        print(f"✓ 有响应! 长度:{len(data)}")
                        print(f"    发送: {packet.hex().upper()}")
                        print(f"    响应: {data.hex().upper()}")

                        responsive_ports.append(port)
                        self.scan_results[port] = {
                            'sent': packet.hex().upper(),
                            'received': data.hex().upper(),
                            'length': len(data)
                        }
                        got_response = True
                        break

                    except socket.timeout:
                        continue
                    except Exception as e:
                        continue

                if not got_response:
                    print("✗ 无响应")

                sock.close()

            except Exception as e:
                print(f"✗ 测试失败: {e}")
                continue

        print(f"\nUDP扫描完成，发现 {len(responsive_ports)} 个响应端口: {responsive_ports}")
        return responsive_ports

    def method3_netstat_scan(self) -> List[int]:
        """方法3: 使用netstat检查MCU开放的端口（如果可以SSH到MCU）"""
        print(f"\n=== 方法3: netstat扫描 ===")
        print("注意: 这需要能够SSH到MCU或在MCU上执行命令")

        # 这里提供SSH命令示例
        ssh_commands = [
            f"ssh root@{self.mcu_ip} 'netstat -an | grep LISTEN'",
            f"ssh root@{self.mcu_ip} 'ss -tuln'",
            f"ssh root@{self.mcu_ip} 'netstat -una'"
        ]

        print("可以尝试的SSH命令:")
        for cmd in ssh_commands:
            print(f"  {cmd}")

        print("\n如果有SSH访问权限，执行上述命令查看开放端口")
        return []

    def method4_nmap_scan(self) -> List[int]:
        """方法4: 使用nmap扫描（如果系统安装了nmap）"""
        print(f"\n=== 方法4: nmap扫描 ===")

        try:
            # 检查是否安装了nmap
            subprocess.run(['nmap', '--version'], capture_output=True, check=True)
            print("发现nmap，开始扫描...")

            # TCP扫描
            cmd_tcp = ['nmap', '-sS', '-p1-10000', '--open', self.mcu_ip]
            print(f"执行: {' '.join(cmd_tcp)}")

            result_tcp = subprocess.run(cmd_tcp, capture_output=True, text=True, timeout=300)

            if result_tcp.returncode == 0:
                print("TCP扫描结果:")
                print(result_tcp.stdout)

            # UDP扫描（常见端口）
            cmd_udp = ['nmap', '-sU', '-p502,1000,1234,5555,5556,8888', self.mcu_ip]
            print(f"执行: {' '.join(cmd_udp)}")

            result_udp = subprocess.run(cmd_udp, capture_output=True, text=True, timeout=120)

            if result_udp.returncode == 0:
                print("UDP扫描结果:")
                print(result_udp.stdout)

        except subprocess.CalledProcessError:
            print("nmap未安装或执行失败")
        except FileNotFoundError:
            print("nmap未安装")
            print("可以安装nmap: ")
            print("  Windows: 下载安装 https://nmap.org/download.html")
            print("  Ubuntu/Debian: sudo apt install nmap")
            print("  CentOS/RHEL: sudo yum install nmap")
        except subprocess.TimeoutExpired:
            print("nmap扫描超时")

        return []

    def method5_wireshark_guide(self):
        """方法5: Wireshark抓包指南"""
        print(f"\n=== 方法5: Wireshark抓包方法 ===")
        print("使用Wireshark抓包是最准确的方法:")
        print()
        print("1. 安装Wireshark")
        print("2. 选择网络接口（连接MCU的网卡）")
        print("3. 设置过滤器:")
        print(f"   ip.addr == {self.mcu_ip}")
        print("4. 开始抓包")
        print("5. 重启MCU或触发MCU通信")
        print("6. 观察MCU发送的数据包，查看源端口")
        print()
        print("关键信息:")
        print("- 查看UDP数据包的源端口")
        print("- 查看TCP连接的端口")
        print("- 注意周期性数据包（可能是数据采集）")
        print()
        print("示例过滤器:")
        print(f"  udp and ip.src == {self.mcu_ip}")
        print(f"  tcp and ip.src == {self.mcu_ip}")

    def method6_ping_and_trace(self):
        """方法6: ping和traceroute测试网络连通性"""
        print(f"\n=== 方法6: 网络连通性测试 ===")

        # Ping测试
        try:
            if sys.platform.startswith('win'):
                ping_cmd = ['ping', '-n', '4', self.mcu_ip]
            else:
                ping_cmd = ['ping', '-c', '4', self.mcu_ip]

            print(f"执行ping测试: {' '.join(ping_cmd)}")
            result = subprocess.run(ping_cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print("✓ Ping成功")
                print(result.stdout)
            else:
                print("✗ Ping失败")
                print(result.stderr)

        except Exception as e:
            print(f"Ping测试失败: {e}")

        # ARP表检查
        try:
            if sys.platform.startswith('win'):
                arp_cmd = ['arp', '-a', self.mcu_ip]
            else:
                arp_cmd = ['arp', '-n', self.mcu_ip]

            print(f"检查ARP表: {' '.join(arp_cmd)}")
            result = subprocess.run(arp_cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                print("ARP表信息:")
                print(result.stdout)
            else:
                print("ARP表中未找到该IP")

        except Exception as e:
            print(f"ARP检查失败: {e}")

    def comprehensive_scan(self, quick_scan: bool = False):
        """综合扫描 - 使用多种方法"""
        print(f"=== MCU端口综合扫描: {self.mcu_ip} ===")

        # 方法6: 网络连通性测试
        self.method6_ping_and_trace()

        # 方法2: UDP响应扫描（最有用）
        responsive_ports = self.method2_udp_response_scan()

        if not quick_scan:
            # 方法1: TCP端口扫描（耗时）
            print("\n是否进行TCP端口扫描？（耗时较长）")
            tcp_scan = input("输入 y 进行TCP扫描，其他键跳过: ").lower().strip()
            if tcp_scan == 'y':
                open_ports = self.method1_tcp_port_scan(range(1, 1000))  # 只扫描前1000个端口

            # 方法4: nmap扫描
            self.method4_nmap_scan()

        # 方法3和5: 指导信息
        self.method3_netstat_scan()
        self.method5_wireshark_guide()

        # 总结
        print(f"\n=== 扫描结果总结 ===")
        if responsive_ports:
            print(f"✓ 发现响应端口: {responsive_ports}")
            print("\n详细响应信息:")
            for port in responsive_ports:
                if port in self.scan_results:
                    info = self.scan_results[port]
                    print(f"  端口 {port}:")
                    print(f"    发送: {info['sent']}")
                    print(f"    响应: {info['received']}")
                    print(f"    长度: {info['length']}")
        else:
            print("✗ 未发现响应端口")

        # 推荐的下一步
        print(f"\n=== 推荐的下一步 ===")
        if responsive_ports:
            print("1. 使用发现的端口号更新XCP客户端代码")
            print("2. 分析响应数据格式，确定是否为XCP协议")
            print("3. 尝试发送标准XCP命令")
        else:
            print("1. 检查MCU是否正在运行")
            print("2. 确认网络配置正确")
            print("3. 使用Wireshark抓包分析")
            print("4. 联系MCU供应商获取端口配置信息")

        return responsive_ports


def main():
    """主函数"""
    print("MCU端口扫描工具")
    print("================")

    # 获取用户输入
    mcu_ip = input("输入MCU IP地址 (默认: 198.18.36.1): ").strip()
    if not mcu_ip:
        mcu_ip = "198.18.36.1"

    local_ip = input("输入本地IP地址 (默认: 198.18.36.100): ").strip()
    if not local_ip:
        local_ip = "198.18.36.100"

    # 创建扫描器
    scanner = MCUPortScanner(mcu_ip, local_ip)

    # 选择扫描模式
    print("\n选择扫描模式:")
    print("1. 快速扫描 (推荐)")
    print("2. 完整扫描")
    print("3. 仅UDP响应扫描")
    print("4. 仅网络连通性测试")

    choice = input("输入选择 (1-4): ").strip()

    try:
        if choice == "1":
            scanner.comprehensive_scan(quick_scan=True)
        elif choice == "2":
            scanner.comprehensive_scan(quick_scan=False)
        elif choice == "3":
            scanner.method2_udp_response_scan()
        elif choice == "4":
            scanner.method6_ping_and_trace()
        else:
            print("使用默认快速扫描")
            scanner.comprehensive_scan(quick_scan=True)

    except KeyboardInterrupt:
        print("\n扫描被用户中断")
    except Exception as e:
        print(f"\n扫描过程中发生错误: {e}")


if __name__ == "__main__":
    main()