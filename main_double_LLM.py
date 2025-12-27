#!/usr/bin/env python3
"""
Product Verification Agent - Simplified Two-Agent Architecture

Architecture:
    - VerificationAgent: 合并的策略+鉴定Agent (原Agent1+Agent3)
        - 任务理解与分解
        - 制定执行计划
        - 协调PhoneAgent
        - 商品鉴定分析
        - 生成报告
    
    - PhoneAgent: 手机操作Agent (原Agent2)
        - 通过AutoGLM控制手机
        - 截图获取
        - 执行搜索、点击等操作

Usage:
    python main.py --query "商品名称" --platform xianyu
    python main.py --query "商品名称" --platform xiaohongshu
    python main.py --interactive

Environment Variables (see .env.example):
    PHONE_AGENT_BASE_URL: AutoGLM API base URL
    PHONE_AGENT_MODEL: AutoGLM model name
    PHONE_AGENT_API_KEY: AutoGLM API key
    PHONE_AGENT_DEVICE_ID: ADB device ID
    LLM_BASE_URL: LLM API base URL (for analysis)
    LLM_MODEL: LLM model name
    LLM_API_KEY: LLM API key
    OUTPUT_DIR: Output directory for results
"""

import os
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# Data Classes
# =============================================================================

class Platform(Enum):
    XIANYU = "xianyu"
    XIAOHONGSHU = "xiaohongshu"


@dataclass
class ProductInfo:
    """商品信息"""
    title: str
    price: float
    platform: str
    description: str = ""
    seller: str = ""
    screenshot_path: Optional[str] = None
    raw_data: Optional[dict] = None


@dataclass
class VerificationResult:
    """鉴定结果"""
    product: ProductInfo
    is_authentic: Optional[bool]
    confidence_score: float
    analysis_summary: str
    risk_indicators: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """鉴定报告"""
    query: str
    platform: str
    timestamp: str
    products_analyzed: int
    authentic_count: int
    suspicious_count: int
    uncertain_count: int
    authenticity_rate: float
    results: List[VerificationResult]
    overall_recommendations: List[str]
    session_dir: str


# =============================================================================
# Phone Agent (Agent2) - 手机操作代理
# =============================================================================

class PhoneAgentWrapper:
    """
    Phone Agent 包装器 - 封装对 main_autoglm 的调用

    职责：
    - 通过ADB控制手机
    - 执行搜索操作
    - 获取截图
    """

    def __init__(self):
        """初始化Phone Agent"""
        self._agent = None
        self._initialized = False

    def _ensure_initialized(self):
        """确保agent已初始化"""
        if self._initialized:
            return

        try:
            from phone_agent import PhoneAgent
            from phone_agent.agent import AgentConfig
            from phone_agent.device_factory import set_device_type
            from phone_agent.model import ModelConfig

            base_url = os.getenv("PHONE_AGENT_BASE_URL",
                                 "http://localhost:8000/v1")
            model = os.getenv("PHONE_AGENT_MODEL", "autoglm-phone-9b")
            api_key = os.getenv("PHONE_AGENT_API_KEY", "EMPTY")
            device_id = os.getenv("PHONE_AGENT_DEVICE_ID")
            max_steps = int(os.getenv("PHONE_AGENT_MAX_STEPS", "100"))

            set_device_type("adb")

            model_config = ModelConfig(
                model_name=model,
                base_url=base_url,
                api_key=api_key,
            )

            agent_config = AgentConfig(
                max_steps=max_steps,
                device_id=device_id,
            )

            self._agent = PhoneAgent(
                model_config=model_config,
                agent_config=agent_config,
            )
            print('-'*50)
            print(f"PhoneAgent initialized with device_id: {device_id}")
            # print(self._agent)
            # print(self._agent.model_config)
            for name in self._agent.model_config.__dict__:
                print(name, self._agent.model_config.__dict__[name])
            # raise ValueError("PhoneAgent initialization failed")
            self._initialized = True

        except Exception as e:
            raise RuntimeError(f"Failed to initialize PhoneAgent: {e}")

    def execute(self, instruction: str) -> str:
        """
        执行手机操作指令

        Args:
            instruction: 自然语言指令

        Returns:
            执行结果
        """
        self._ensure_initialized()
        try:
            result = self._agent.run(instruction)
            return str(result) if result else "操作完成"
        except Exception as e:
            return f"操作失败: {str(e)}"

    def reset(self):
        """重置agent状态"""
        if self._agent:
            self._agent.reset()

# =============================================================================
# Verification Agent (Agent1 + Agent3 合并) - 策略+鉴定代理
# =============================================================================

class VerificationAgent:
    """
    双模型协作商品鉴定系统

    架构设计：
    - 总指挥 (deepseek): 负责思考、分析、决策、下达指令
    - 执行器 (autoglm-phone): 负责执行手机操作，返回结果

    协作方式：
    - 初始：deepseek 收到任务提示词
    - 循环：
      1. deepseek 分析当前状态，思考下一步应该做什么
      2. 生成 instruction 发送给 autoglm-phone
      3. autoglm-phone 执行操作，返回结果
      4. 结果返回给 deepseek，继续分析
    - 结束：deepseek 判断任务完成，生成最终报告
    """
    def __init__(self, phone_agent: PhoneAgentWrapper, folder_output: str):
        """
        初始化 VerificationAgent_test

        Args:
            phone_agent: 手机操作代理 (autoglm-phone)
        """
        self.phone_agent = phone_agent
        self.folder_output = folder_output
        

        self.llm_client = self._create_llm_client()
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")
        self.temperature = 0.7
        self.max_tokens = 500

        self.message_history: list[dict] = []
        from screenshot import take_screenshot
        self.take_screenshot = take_screenshot

    def _create_llm_client(self) -> Optional[object]:
        """创建LLM客户端"""
        try:
            base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
            api_key = os.getenv("LLM_API_KEY")
            model = os.getenv("LLM_MODEL", "deepseek-chat")

            if not api_key:
                raise ValueError("LLM_API_KEY 未配置")

            return OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            print(f"  ⚠️ LLM客户端初始化失败: {e}")
            return None

    def run(
        self,
        query: str,
        platform: Platform,
        max_products: int = 5
    ) -> dict:
        """
        运行双模型协作鉴定流程

        流程由总指挥模型自主决定，核心循环：
        1. 总指挥分析状态，生成指令
        2. 执行器执行手机操作
        3. 结果返回总指挥
        4. 总指挥判断是否继续或结束

        Args:
            query: 搜索关键词
            platform: 平台 (xianyu/xiaohongshu)
            max_products: 最大商品数

        Returns:
            鉴定报告字典
        """
        platform_str = platform.value if isinstance(
            platform, Platform) else platform
        platform_name = "闲鱼" if platform_str == "xianyu" else "小红书"

        self._log_header(query, platform_name, max_products)

        self._init_message_history(query, platform_name, max_products)

        self._log("\n" + "="*50)
        self._log("  双模型协作开始")
        self._log("  总指挥: deepseek | 执行器: autoglm-phone")
        self._log("="*50)

        max_iterations = 15
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self._log(f"\n[迭代 {iteration}/{max_iterations}]")

            commander_response = self._get_commander_response()

            if not commander_response:
                self._log("  ⚠️ 总指挥未能生成有效响应")
                break

            self._log(f"  📤 总指挥响应: {commander_response[:100]}...")
            self.message_history.append(
                {"role": "assistant", "content": commander_response})

            if self._is_task_complete(commander_response):
                self._log("  ✅ 总指挥判定任务完成")
                break

            instruction = self._extract_instruction_from_code_block(
                commander_response)

            if not instruction:
                self._log("  ⚠️ 未能从总指挥响应中提取有效指令(需要```...```包裹)")
                continue

            if "截图" == instruction:
                self.take_screenshot(path=self.folder_output)
                phone_result = "截图已保存在" + self.folder_output
            else:
                self._log(f"  📱 执行器指令: {instruction[:80]}...")
                phone_result = self._execute_phone_instruction(instruction)

            self._log(f"  📥 执行器响应: {phone_result[:80]}...")
            self.message_history.append(
                {"role": "user", "content": phone_result})

        final_report = self._generate_final_report()

        self._log("\n" + "="*50)
        self._log("  鉴定完成")
        self._log("="*50)

        return final_report

    def _init_message_history(
        self,
        query: str,
        platform_name: str,
        max_products: int
    ):
        """初始化消息历史，添加 system 提示词和初始任务"""
        from phone_agent.prompt import 基础提示词 as prompt
        system_prompt = prompt + f"""当前任务：
        - 搜索关键词: {query}
        - 目标平台: {platform_name}
        - 鉴定数量: 前{max_products}个商品

        请开始分析，指令必须用 ```...``` 包裹。"""

        self.message_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请鉴定{platform_name}上'{query}'相关的商品是否为正品。第一步指令必须用 ```...``` 包裹。"}
        ]

    def _get_commander_response(self) -> Optional[str]:
        """获取总指挥的响应"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=self.message_history,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            # print(f"deepseek 指挥: {response}")

            content = response.choices[0].message.content.strip()
            return content if content else None

        except Exception as e:
            self._log(f"  ⚠️ 获取总指挥响应失败: {e}")
            return None

    def _execute_phone_instruction(self, instruction: str) -> str:
        """执行器执行手机指令并处理结果"""
        try:
            raw_result = self.phone_agent.execute(instruction)
            result = raw_result if raw_result else ""

            if not result:
                return "执行完成，无返回信息"

            result = result.strip()

            error_indicators = ["无法", "失败", "错误", "异常", "找不到", "未找到"]
            for indicator in error_indicators:
                if indicator in result and len(result) < 100:
                    return f"执行结果: {result}"

            lines = [l.strip() for l in result.split('\n') if l.strip()]
            if len(lines) > 1:
                summary = f"已获取{len(lines)}条信息"
                if lines:
                    first_info = lines[0][:50]
                    summary += f"，首条: {first_info}..."
                return f"{summary}\n\n详细信息:\n{result}"

            if len(result) > 200:
                return f"{result[:200]}...\n(更多内容已截断)"

            return result

        except Exception as e:
            return f"执行失败: {str(e)}"

    def _parse_phone_result(self, result: str) -> dict:
        """解析执行器返回的结果"""
        parsed = {
            "raw": result,
            "has_content": bool(result and result.strip()),
            "error": False,
            "price": None,
            "title": None
        }

        error_indicators = ["无法", "失败", "错误", "异常", "找不到"]
        for indicator in error_indicators:
            if indicator in result:
                parsed["error"] = True
                break

        price_patterns = [
            r'[¥￥](\d+\.?\d*)',
            r'价格[：:]\s*(\d+\.?\d*)',
            r'¥(\d+\.?\d*)'
        ]
        for pattern in price_patterns:
            match = re.search(pattern, result)
            if match:
                try:
                    parsed["price"] = float(match.group(1))
                    break
                except ValueError:
                    pass

        lines = [l.strip() for l in result.split('\n') if l.strip()]
        if lines:
            for line in lines[:3]:
                if len(line) > 5 and len(line) < 100:
                    if not line.startswith(('执行', '结果', '已获', '请')):
                        parsed["title"] = line[:50]
                        break

        return parsed

    def _format_result_for_commander(self, parsed_result: dict) -> str:
        """将解析后的结果格式化为适合总指挥理解的格式"""
        parts = []

        if parsed_result["error"]:
            parts.append("⚠️ 执行遇到问题")

        if parsed_result["title"]:
            parts.append(f"📦 商品: {parsed_result['title']}")

        if parsed_result["price"]:
            parts.append(f"💰 价格: ¥{parsed_result['price']}")

        if not parts:
            raw = parsed_result["raw"]
            if len(raw) > 100:
                parts.append(f"📋 结果摘要: {raw[:100]}...")
            else:
                parts.append(f"📋 结果: {raw}")

        return " | ".join(parts)

    def _is_task_complete(self, last_message: str) -> bool:
        """判断任务是否完成"""
        complete_keywords = [
            "鉴定完成", "任务完成", "最终结论", "报告生成", "最终报告",
            "已完成", "全部完成", "鉴定结束", "分析完毕", "全部鉴定", "所有商品鉴定完成"
        ]

        for keyword in complete_keywords:
            if keyword in last_message:
                return True

        return False

    def _extract_instruction_from_code_block(self, content: str) -> Optional[str]:
        """使用正则匹配来提取代码块中的指令"""
        if not content:
            return None

        content = content.strip()
        code_block_pattern = r'```(?:\w*\s*)?\n?([\s\S]*?)\n?```'
        matches = re.findall(code_block_pattern, content)

        if matches:
            instruction = matches[0].strip()
            if instruction:
                return instruction
        return None

    def _generate_final_report(self) -> dict:
        """生成最终报告"""
        os.makedirs(self.folder_output, exist_ok=True)

        try:
            self.message_history.append({
                "role": "user",
                "content": "请生成最终的商品鉴定报告，包含：商品信息、真伪判断、置信度、风险指标、购买建议。以JSON格式返回，key包括：query, platform, products_analyzed, authentic_count, suspicious_count, uncertain_count, authenticity_rate, results(数组，每个包含title, price, is_authentic, confidence, analysis, risk_indicators, recommendations), overall_recommendations"
            })

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=self.message_history,
                temperature=0.3,
                max_tokens=5000
            )

            content = response.choices[0].message.content.strip()

            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            report = json.loads(content)

            report_data = {
                "query": report.get("query", ""),
                "platform": report.get("platform", ""),
                "marketplace_products": [p.get("title", "") for p in report.get("results", [])],
                "authentic_reference": {},
                "comparison": {
                    "summary": f"已分析 {report.get('products_analyzed', 0)} 个商品",
                    "total_products": report.get("products_analyzed", 0),
                    "authentic_count": report.get("authentic_count", 0),
                    "suspicious_count": report.get("suspicious_count", 0),
                    "uncertain_count": report.get("uncertain_count", 0),
                    "authenticity_rate": report.get("authenticity_rate", 0)
                },
                "results": report.get("results", []),
                "recommendations": report.get("overall_recommendations", [])
            }

            report_path = os.path.join(self.folder_output, "final_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            self._log(f"  📄 报告已保存: {report_path}")

            return report_data

        except Exception as e:
            self._log(f"  ⚠️ 生成报告失败: {e}")
            return self._generate_simple_report()

    def _generate_simple_report(self) -> dict:
        """生成简单报告（当LLM生成失败时）"""
        return {
            "query": "待生成",
            "platform": "待生成",
            "marketplace_products": [],
            "authentic_reference": {},
            "comparison": {"summary": "报告生成失败，请手动分析"},
            "results": [],
            "recommendations": ["请检查LLM配置后重试"]
        }

    def _log_header(self, query: str, platform: str, max_products: int):
        """输出头部信息"""
        print("\n" + "=" * 60)
        print("  双模型协作鉴定系统 - VerificationAgent_test")
        print("  Architecture: deepseek (总指挥) + autoglm-phone (执行器)")
        print("=" * 60)
        print(f"  搜索关键词: {query}")
        print(f"  目标平台: {platform}")
        print(f"  最大商品数: {max_products}")
        print("=" * 60)

    def _log(self, message: str):
        """日志输出"""
        print(message)

# =============================================================================
# Main Functions
# =============================================================================

def main():
    load_dotenv()

    print("\n" + "=" * 60)
    print("  VerificationAgent_test 测试")
    print("  双模型协作: deepseek(总指挥) + autoglm-phone(执行器)")
    print("=" * 60)

    phone_agent = PhoneAgentWrapper()

    query = "刘勃讲中国史"
    platform = Platform.XIANYU
    max_products = 2

    print(f"\n测试参数:")
    print(f"  搜索关键词: {query}")
    print(f"  目标平台: {'闲鱼' if platform == Platform.XIANYU else '小红书'}")
    print(f"  最大商品数: {max_products}")
    print("=" * 60)
    folder_output=os.path.join(
            "output", f"{platform}_{query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    agent = VerificationAgent(
        phone_agent=phone_agent,
        folder_output=folder_output
    )

    try:
        report = agent.run(
            query=query,
            platform=platform,
            max_products=max_products
        )

        print("\n" + "=" * 60)
        print("  测试完成 - 报告摘要")
        print("=" * 60)
        print(f"  查询关键词: {report.get('query', 'N/A')}")
        print(f"  平台: {report.get('platform', 'N/A')}")
        print(f"  对比结果: {report.get('comparison', {}).get('summary', 'N/A')}")
        print(
            f"  正品率: {report.get('comparison', {}).get('authenticity_rate', 'N/A')}")
        print(f"  建议数量: {len(report.get('recommendations', []))}")
        print("=" * 60)

        return report

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

