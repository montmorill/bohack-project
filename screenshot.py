# 利用ADB抓取屏幕截图的Python程序
import subprocess
import os
from datetime import datetime

def find_adb():
    """查找ADB可执行文件路径"""
    # 使用Android Studio默认的ADB路径
    default_adb_path = r"C:\Users\14048\AppData\Local\Android\Sdk\platform-tools\adb.exe"
    if os.path.exists(default_adb_path):
        return default_adb_path
    return "adb"  # 如果已添加到PATH

def take_screenshot():
    """使用ADB抓取设备屏幕截图"""
    print("=== ADB屏幕截图工具 ===")
    
    try:
        adb_path = find_adb()
        
        # 1. 检查设备连接情况
        print("🔍 检查设备连接...")
        result = subprocess.run([adb_path, "devices"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"❌ 执行adb命令失败: {result.stderr}")
            return 1
        
        # 解析设备列表
        device_lines = [line for line in result.stdout.split('\n') if 'device' in line and not line.startswith('List')]
        if not device_lines:
            print("❌ 未发现任何设备，请确保设备已连接并开启USB调试")
            return 1
        
        device_serial = device_lines[0].split('\t')[0]
        print(f"✓ 发现设备: {device_serial}")
        
        # 2. 生成截图文件名和路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_screenshot_path = f"screenshot_test.png"
        device_screenshot_path = f"/sdcard/screenshot_test.png"
        
        # 3. 在设备上抓取截图
        print("📸 正在抓取屏幕截图...")
        screencap_cmd = [adb_path, "-s", device_serial, "shell", "screencap", "-p", device_screenshot_path]
        result = subprocess.run(screencap_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            print(f"❌ 在设备上抓取截图失败: {result.stderr}")
            return 1
        
        # 4. 将截图从设备传输到本地
        print("📤 正在传输截图到本地...")
        pull_cmd = [adb_path, "-s", device_serial, "pull", device_screenshot_path, local_screenshot_path]
        result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            print(f"❌ 传输截图失败: {result.stderr}")
            return 1
        
        # 5. 删除设备上的临时截图
        subprocess.run([adb_path, "-s", device_serial, "shell", "rm", device_screenshot_path], 
                      capture_output=True, text=True, timeout=10)
        
        print(f"✅ 截图已成功保存到: {local_screenshot_path}")
        print(f"📁 文件位置: {os.path.abspath(local_screenshot_path)}")
        
        return 0
        
    except subprocess.TimeoutExpired:
        print("❌ 命令执行超时")
        return 1
    except Exception as e:
        print(f"❌ 抓取截图失败: {e}")
        print("请检查：")
        print("1. ADB服务器是否运行 (adb start-server)")
        print("2. Android设备/模拟器是否连接")
        print("3. 设备是否开启了USB调试")
        return 1

def main():
    """主函数"""
    return take_screenshot()

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

