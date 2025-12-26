# 利用ADB安装APK到手机的Python程序
import subprocess
import os
from datetime import datetime

def find_adb():
    """查找ADB可执行文件路径"""
    # 使用Android Studio默认的ADB路径
    default_adb_path = r"C:\Users\14048\AppData\Local\Android\Sdk\platform-tools\adb.exe"
    if os.path.exists(default_adb_path):
        return default_adb_path
    return "adb"  # 如果已添加到 PATH

def install_apk():
    """使用ADB将APK文件安装到设备"""
    print("=== ADB APK安装工具 ===")
    
    try:
        adb_path = find_adb()
        
        # 1. 检查设备连接情况
        print("🔍 检查设备连接...")
        result = subprocess.run([adb_path, "devices"], capture_output=True, text=True, encoding='utf-8', timeout=10)
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
        
        # 获取设备架构信息
        print("🔍 检查设备架构...")
        abi_result = subprocess.run([adb_path, "-s", device_serial, "shell", "getprop", "ro.product.cpu.abi"], 
                                  capture_output=True, text=True, encoding='utf-8', timeout=10)
        device_abi = abi_result.stdout.strip()
        print(f"✓ 设备架构: {device_abi}")
        
        # 获取设备支持的所有架构
        abi_list_result = subprocess.run([adb_path, "-s", device_serial, "shell", "getprop", "ro.product.cpu.abilist"], 
                                      capture_output=True, text=True, encoding='utf-8', timeout=10)
        device_abi_list = abi_list_result.stdout.strip().split(',')
        print(f"✓ 设备支持的所有架构: {device_abi_list}")
        
        # 2. 检查APK文件夹
        apk_folder = "./apk"
        if not os.path.exists(apk_folder):
            print(f"❌ APK文件夹不存在: {apk_folder}")
            return 1
        
        # 获取所有APK文件
        apk_files = [f for f in os.listdir(apk_folder) if f.endswith('.apk')]
        if not apk_files:
            print(f"❌ APK文件夹中未找到APK文件: {apk_folder}")
            return 1
        
        print(f"✓ 发现 {len(apk_files)} 个APK文件")
        
        # 3. 安装每个APK文件
        for apk_file in apk_files:
            local_apk_path = os.path.join(apk_folder, apk_file)
            
            print(f"\n📦 处理APK: {apk_file}")
            print(f"📁 本地路径: {local_apk_path}")
            
            # 尝试默认安装
            print("🔧 正在安装APK...")
            install_cmd = [adb_path, "-s", device_serial, "install", "-r", local_apk_path]
            result = subprocess.run(install_cmd, capture_output=True, text=True, encoding='utf-8', timeout=60)
            
            if result.returncode == 0:
                print(f"✅ 成功安装: {apk_file}")
            else:
                print(f"❌ 安装APK失败: {result.stderr}")
                
                # 检查是否是架构不匹配问题
                if "INSTALL_FAILED_NO_MATCHING_ABIS" in result.stderr:
                    print("💡 问题分析: 架构不匹配，APK可能只支持ARM架构，而设备是x86/x86_64架构")
                    print("🔧 尝试解决方案1: 使用--abi选项强制指定架构")
                    
                    # 尝试使用--abi选项强制安装
                    architectures = ["x86", "x86_64", "armeabi-v7a", "arm64-v8a"]
                    install_success = False
                    
                    for arch in architectures:
                        print(f"\n🔄 尝试使用架构 {arch} 安装...")
                        force_install_cmd = [adb_path, "-s", device_serial, "install", "-r", "--abi", arch, local_apk_path]
                        force_result = subprocess.run(force_install_cmd, capture_output=True, text=True, encoding='utf-8', timeout=60)
                        
                        if force_result.returncode == 0:
                            print(f"✅ 成功安装: {apk_file} (使用架构 {arch})")
                            install_success = True
                            break
                        else:
                            print(f"❌ 使用架构 {arch} 安装失败: {force_result.stderr}")
                    
                    if not install_success:
                        print("\n📋 建议解决方案:")
                        print("1. 对于模拟器: 创建ARM架构的模拟器或启用ARM兼容层")
                        print("2. 对于真机: 确保下载的APK支持设备的CPU架构")
                        print("3. 尝试使用其他来源的APK文件，确保支持x86/x86_64架构")
                        print("4. 检查设备是否已开启'允许安装未知来源应用'选项")
        
        print("\n🎉 所有APK文件处理完成！")
        return 0
        
    except subprocess.TimeoutExpired:
        print("❌ 命令执行超时")
        return 1
    except Exception as e:
        print(f"❌ 安装APK失败: {e}")
        print("请检查：")
        print("1. ADB服务器是否运行 (adb start-server)")
        print("2. Android设备/模拟器是否连接")
        print("3. 设备是否开启了USB调试")
        return 1

def main():
    """主函数"""
    return install_apk()

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)