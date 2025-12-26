# 检查APK支持的CPU架构
import subprocess
import os

def main():
    aapt_path = r'C:\Users\14048\AppData\Local\Android\Sdk\build-tools\36.1.0\aapt.exe'
    apk_path = os.path.join('.', 'apk', '小红书.apk')
    
    print(f"使用AAPT路径: {aapt_path}")
    print(f"检查APK: {apk_path}")
    
    try:
        result = subprocess.run([aapt_path, 'dump', 'badging', apk_path], 
                              capture_output=True, 
                              text=True, 
                              encoding='utf-8', 
                              errors='ignore',
                              timeout=30)
        
        print("\nAPK信息:")
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('package:'):
                print(line)
            elif 'native-code' in line:
                print(line)
            elif 'sdkVersion' in line:
                print(line)
            elif 'targetSdkVersion' in line:
                print(line)
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()