#!/usr/bin/env python3
"""
交互式手机操控Agent
流程: 用户输入任务 → Agent1(DeepSeek)生成步骤 → Agent2执行操作
"""

import argparse
import subprocess
import sys

from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.model import ModelConfig
from phone_agent.config.prompts_piracy import PromptGenerator, PromptGeneratorConfig


def check_adb(device_id: str = None) -> bool:
    """检查ADB环境和设备连接"""
    try:
        r = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10)
        devices = [l for l in r.stdout.split("\n")[1:] if "\tdevice" in l]
        if not devices:
            print("❌ 无设备连接，请先运行: adb connect <ip>:<port>")
            return False
        print(f"✅ 设备已连接: {devices[0].split()[0]}")
        return True
    except Exception:
        print("❌ ADB检查失败")
        return False


def connect_device(address: str) -> bool:
    """连接ADB设备"""
    print(f"📱 连接设备 {address}...")
    try:
        r = subprocess.run(["adb", "connect", address], capture_output=True, text=True, timeout=10)
        if "connected" in r.stdout.lower() or "already" in r.stdout.lower():
            print(f"✅ 已连接")
            return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="交互式手机操控Agent")
    parser.add_argument("--base-url", default="https://open.bigmodel.cn/api/paas/v4", help="Agent2模型API地址")
    parser.add_argument("--model", default="autoglm-phone", help="Agent2模型名称")
    parser.add_argument("--apikey", default="EMPTY", help="Agent2 API Key")
    parser.add_argument("--device-id", "-d", help="设备ID")
    parser.add_argument("--connect", "-c", help="连接设备地址 如127.0.0.1:5555")
    parser.add_argument("--deepseek-key", default=None, help="DeepSeek API Key (Agent1)")
    args = parser.parse_args()
    
    # 连接设备
    if args.connect:
        connect_device(args.connect)
        args.device_id = args.connect
    
    # 检查ADB
    if not check_adb(args.device_id):
        sys.exit(1)
    
    # 初始化Agent1 (DeepSeek提示词生成器)
    agent1_config = PromptGeneratorConfig()
    if args.deepseek_key:
        agent1_config.api_key = args.deepseek_key
    prompt_generator = PromptGenerator(agent1_config)
    
    # Agent2模型配置
    model_config = ModelConfig(
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.apikey
    )
    
    print("\n" + "=" * 60)
    print("🤖 交互式手机操控Agent")
    print("=" * 60)
    print("Agent1: DeepSeek (任务分解与提示词生成)")
    print("Agent2: AutoGLM (手机操控执行)")
    print("=" * 60)
    print("输入任务描述，Agent1会生成详细步骤，Agent2会执行操作")
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 60)
    
    while True:
        print("\n")
        task = input("📝 请输入要完成的任务: ").strip()
        
        if not task:
            print("⚠️ 请输入有效的任务描述")
            continue
        
        if task.lower() in ['quit', 'exit', 'q']:
            print("👋 再见!")
            break
        
        # 可选：输入平台信息
        platform = input("📱 目标平台(可选，直接回车跳过): ").strip()
        
        try:
            # ========== Agent1: 生成提示词 ==========
            system_prompt = prompt_generator.generate_prompt(
                user_task=task,
                platform=platform
            )
            
            # ========== Agent2: 执行任务 ==========
            print("\n" + "=" * 60)
            print("🚀 Agent2 (AutoGLM) 开始执行任务...")
            print("=" * 60)
            
            agent2 = PhoneAgent(
                model_config=model_config,
                agent_config=AgentConfig(
                    max_steps=50,
                    device_id=args.device_id,
                    lang="cn",
                    verbose=True,
                    system_prompt=system_prompt
                )
            )
            
            # 执行任务
            result = agent2.run(task)
            
            print("\n" + "=" * 60)
            print(f"✅ 任务完成: {result}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n⚠️ 任务被中断")
        except Exception as e:
            print(f"\n❌ 执行出错: {e}")


if __name__ == "__main__":
    main()
