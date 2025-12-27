#!/usr/bin/env python3
"""
得到盗版内容检测Agent
在闲鱼/小红书上自动检测和举报盗版"得到"App内容

流程: 得到App确认正版 → 目标平台搜索 → 逐个检测商品 → 生成报告
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openai import OpenAI
from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.adb import get_screenshot
from phone_agent.model import ModelConfig
from phone_agent.config.prompts_piracy import get_piracy_detection_prompt


# ==================== 数据结构 ====================

@dataclass
class ContentInfo:
    """商品/笔记内容信息"""
    title: str = ""
    description: str = ""
    price: str = ""
    comments: list[str] = field(default_factory=list)
    seller_info: str = ""
    platform: str = ""
    screenshot_base64: str = ""


@dataclass
class PiracyResult:
    """盗版分析结果"""
    is_piracy: bool
    confidence: float
    reasoning: str
    evidence: list[str] = field(default_factory=list)
    risk_level: str = "low"  # low/medium/high


@dataclass
class DetectionRecord:
    """检测记录"""
    timestamp: str
    platform: str
    content: ContentInfo
    result: PiracyResult
    reported: bool = False


# ==================== 环境检查 ====================

def check_adb(device_id: str = None) -> bool:
    """检查ADB环境和设备连接"""
    print("🔍 检查系统环境...")
    
    # 检查ADB
    try:
        r = subprocess.run(["adb", "version"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print("❌ ADB未安装")
            return False
    except Exception:
        print("❌ ADB未安装")
        return False
    
    # 检查设备
    try:
        r = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10)
        devices = [l for l in r.stdout.split("\n")[1:] if "\tdevice" in l]
        if not devices:
            print("❌ 无设备连接，请先运行: adb connect <ip>:<port>")
            return False
        print(f"✅ 设备已连接: {devices[0].split()[0]}")
    except Exception:
        print("❌ 检查设备失败")
        return False
    
    return True


def check_api(base_url: str, model: str, api_key: str) -> bool:
    """检查模型API连接"""
    print(f"🔍 检查API连接...")
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30)
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5
        )
        if r.choices:
            print("✅ API连接正常")
            return True
    except Exception as e:
        print(f"❌ API连接失败: {e}")
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


# ==================== 内容分析器 ====================

class ContentAnalyzer:
    """多模态内容分析器 - 分析截图内容并判断是否盗版"""
    
    # 盗版特征关键词
    PIRACY_KEYWORDS = [
        "网盘", "百度云", "阿里云盘", "夸克", "链接", "提取码",
        "电子版", "PDF", "epub", "音频", "MP3", "视频",
        "打包", "全集", "合集", "资源", "私发", "秒发"
    ]
    
    # 正版二手特征
    LEGIT_KEYWORDS = [
        "二手", "闲置", "转让", "实体书", "纸质", "正版",
        "九成新", "八成新", "包邮", "自提"
    ]
    
    def __init__(self, model_config: ModelConfig):
        self.client = OpenAI(
            base_url=model_config.base_url,
            api_key=model_config.api_key
        )
        self.model = model_config.model_name
    
    def analyze_screenshot(self, img_base64: str) -> dict:
        """分析截图提取商品信息"""
        prompt = """分析截图，提取商品信息，返回JSON：
{"title": "标题", "description": "描述", "price": "价格", "seller": "卖家", "comments": ["评论1"]}"""
        
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=1500,
                temperature=0.1
            )
            content = r.choices[0].message.content
            # 提取JSON
            start, end = content.find('{'), content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception as e:
            print(f"  分析失败: {e}")
        return {}
    
    def check_piracy(self, info: ContentInfo) -> PiracyResult:
        """判断是否为盗版"""
        text = f"{info.title} {info.description} {' '.join(info.comments)}".lower()
        
        # 统计特征
        piracy_found = [k for k in self.PIRACY_KEYWORDS if k in text]
        legit_found = [k for k in self.LEGIT_KEYWORDS if k in text]
        
        # 判断逻辑
        evidence = []
        if piracy_found:
            evidence.append(f"盗版特征: {', '.join(piracy_found)}")
        if legit_found:
            evidence.append(f"正版特征: {', '.join(legit_found)}")
        
        # 核心判断
        if piracy_found and not legit_found:
            return PiracyResult(True, min(0.5 + len(piracy_found)*0.1, 0.95), 
                              f"检测到{len(piracy_found)}项盗版特征", evidence, "high")
        elif piracy_found and legit_found:
            is_piracy = len(piracy_found) > len(legit_found)
            return PiracyResult(is_piracy, 0.5, "特征混合，需人工确认", evidence, "medium")
        elif legit_found:
            return PiracyResult(False, 0.8, "正版二手商品", evidence, "low")
        else:
            return PiracyResult(False, 0.3, "未检测到明显特征", evidence, "low")


# ==================== 主检测Agent ====================

class PiracyDetector:
    """盗版检测Agent"""
    
    PLATFORM_CONFIG = {
        "xianyu": {"app": "闲鱼", "has_tab": False},
        "xiaohongshu": {"app": "小红书", "has_tab": True}  # 需点击"商品"标签
    }
    
    def __init__(self, model_config: ModelConfig, platform: str, 
                 max_items: int = 10, device_id: str = None, auto_report: bool = False):
        self.platform = platform
        self.max_items = max_items
        self.device_id = device_id
        self.auto_report = auto_report
        self.config = self.PLATFORM_CONFIG.get(platform, self.PLATFORM_CONFIG["xianyu"])
        
        # 初始化组件
        self.analyzer = ContentAnalyzer(model_config)
        self.agent = PhoneAgent(
            model_config=model_config,
            agent_config=AgentConfig(
                max_steps=100, device_id=device_id, lang="cn", verbose=True,
                system_prompt=get_piracy_detection_prompt(platform)
            )
        )
        
        # 检测结果
        self.records: list[DetectionRecord] = []
        self.checked = 0
        self.reported = 0
    
    def run(self, keyword: str) -> list[DetectionRecord]:
        """运行检测流程"""
        app = self.config["app"]
        print("=" * 60)
        print(f"🔍 得到盗版检测 | 平台: {app} | 关键词: {keyword}")
        print("=" * 60)
        
        try:
            # 1. 得到App确认正版
            self._step1_confirm_official(keyword)
            
            # 2. 目标平台搜索
            self._step2_search(keyword)
            
            # 3. 循环检测
            while self.checked < self.max_items:
                print(f"\n[{self.checked+1}/{self.max_items}] 检测中...")
                self._step3_check_item()
                self.checked += 1
                if self.checked < self.max_items:
                    self._next_item()
                    
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
        finally:
            self._save_report()
        
        return self.records
    
    def _step1_confirm_official(self, keyword: str):
        """第一步: 在得到App确认正版产品"""
        print("\n" + "=" * 50)
        print("📖 第一步: 得到App确认正版")
        print("=" * 50)
        
        task = f"启动得到，搜索'{keyword}'，点击第一个结果查看详情"
        print(f"执行: {task}")
        
        try:
            self.agent.run(task)
            self.agent.reset()
            time.sleep(2)
            print("\n请查看模拟器中的正版产品信息")
            input("确认后按回车继续...")
        except Exception as e:
            print(f"⚠️ {e}")
    
    def _step2_search(self, keyword: str):
        """第二步: 在目标平台搜索"""
        app = self.config["app"]
        print("\n" + "=" * 50)
        print(f"📱 第二步: {app}搜索")
        print("=" * 50)
        
        # 启动应用
        self.agent.run(f"启动{app}")
        self.agent.reset()
        time.sleep(3)
        
        # 检查登录
        self._check_login(app)
        
        # 搜索
        if self.config["has_tab"]:
            task = f"搜索'{keyword}'，然后点击'商品'标签筛选"
        else:
            task = f"搜索'{keyword}'"
        
        print(f"执行: {task}")
        self.agent.run(task)
        self.agent.reset()
        time.sleep(2)
    
    def _check_login(self, app: str):
        """检查是否需要登录"""
        try:
            screenshot = get_screenshot(self.device_id)
            analysis = self.analyzer.analyze_screenshot(screenshot.base64_data)
            text = str(analysis).lower()
            
            if any(k in text for k in ["登录", "登陆", "验证码", "手机号"]):
                print(f"\n⚠️ {app}需要登录，请手动完成后按回车...")
                input()
        except:
            pass
    
    def _step3_check_item(self):
        """第三步: 检测单个商品"""
        try:
            # 进入详情
            self.agent.run("点击第一个搜索结果进入详情")
            self.agent.reset()
            time.sleep(2)
            
            # 截图分析
            screenshot = get_screenshot(self.device_id)
            info = ContentInfo(platform=self.platform, screenshot_base64=screenshot.base64_data)
            
            data = self.analyzer.analyze_screenshot(screenshot.base64_data)
            if data:
                info.title = data.get("title", "")
                info.description = data.get("description", "")
                info.price = data.get("price", "")
                info.seller_info = data.get("seller", "")
                info.comments = data.get("comments", [])
            
            # 判断盗版
            result = self.analyzer.check_piracy(info)
            self._print_result(info, result)
            
            # 记录
            record = DetectionRecord(
                timestamp=datetime.now().isoformat(),
                platform=self.platform,
                content=info,
                result=result
            )
            
            # 举报处理
            if result.is_piracy and result.confidence >= 0.6:
                if self.auto_report or input("举报? (y/n): ").lower() == 'y':
                    self._do_report()
                    record.reported = True
                    self.reported += 1
            
            self.records.append(record)
            
        except Exception as e:
            print(f"检测失败: {e}")
    
    def _print_result(self, info: ContentInfo, result: PiracyResult):
        """打印分析结果"""
        print(f"\n{'='*40}")
        print(f"标题: {info.title[:30]}..." if len(info.title) > 30 else f"标题: {info.title}")
        print(f"价格: {info.price}")
        print(f"盗版: {'是⚠️' if result.is_piracy else '否✅'} | 置信度: {result.confidence:.0%}")
        print(f"理由: {result.reasoning}")
        print(f"{'='*40}")
    
    def _do_report(self):
        """执行举报"""
        print("🚨 举报中...")
        self.agent.run("点击举报，选择侵权")
        self.agent.reset()
        time.sleep(1)
    
    def _next_item(self):
        """准备下一个商品"""
        self.agent.run("返回，向下滑动查看更多")
        self.agent.reset()
        time.sleep(1.5)
    
    def _save_report(self):
        """保存检测报告"""
        if not self.records:
            return
        
        filename = f"report_{self.platform}_{datetime.now():%Y%m%d_%H%M%S}.json"
        piracy_count = sum(1 for r in self.records if r.result.is_piracy)
        
        report = {
            "summary": {
                "platform": self.platform,
                "checked": self.checked,
                "piracy_found": piracy_count,
                "reported": self.reported
            },
            "records": [{
                "time": r.timestamp,
                "title": r.content.title,
                "price": r.content.price,
                "is_piracy": r.result.is_piracy,
                "confidence": r.result.confidence,
                "reasoning": r.result.reasoning,
                "reported": r.reported
            } for r in self.records]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 报告: {filename}")
        print(f"📊 检测:{self.checked} | 盗版:{piracy_count} | 举报:{self.reported}")


# ==================== 入口 ====================

def select_platform() -> str:
    """选择平台"""
    print("\n选择平台: 1.闲鱼  2.小红书")
    while True:
        c = input("输入(1/2): ").strip()
        if c == "1": return "xianyu"
        if c == "2": return "xiaohongshu"


def main():
    parser = argparse.ArgumentParser(description="得到盗版检测Agent")
    parser.add_argument("--platform", choices=["xianyu", "xiaohongshu"])
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--auto-report", action="store_true")
    parser.add_argument("--base-url", default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--model", default="autoglm-phone")
    parser.add_argument("--apikey", default="EMPTY")
    parser.add_argument("--device-id", "-d")
    parser.add_argument("--connect", "-c", help="连接设备 如127.0.0.1:5555")
    parser.add_argument("--keyword", "-k", default="薛兆丰漫画经济学")
    args = parser.parse_args()
    
    # 选择平台
    platform = args.platform or select_platform()
    print(f"\n✅ 平台: {'闲鱼' if platform == 'xianyu' else '小红书'}")
    
    # 连接设备
    if args.connect:
        connect_device(args.connect)
        args.device_id = args.connect
    
    # 环境检查
    if not check_adb(args.device_id):
        sys.exit(1)
    if not check_api(args.base_url, args.model, args.apikey):
        sys.exit(1)
    
    # 运行检测
    detector = PiracyDetector(
        model_config=ModelConfig(
            base_url=args.base_url,
            model_name=args.model,
            api_key=args.apikey
        ),
        platform=platform,
        max_items=args.max_items,
        device_id=args.device_id,
        auto_report=args.auto_report
    )
    detector.run(args.keyword)


if __name__ == "__main__":
    main()
