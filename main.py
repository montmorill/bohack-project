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

import argparse
import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List
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

class VerificationAgent_test:
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
        system_prompt = f"""你是一个专业的商品真伪鉴定专家，负责鉴定二手平台商品是否为正品。

你的任务：对比闲鱼（或小红书）上的商品与得到APP上的正品，判断闲鱼（或小红书）中的商品是否为正品。

工作方式：
1. 你需要自主思考并决定下一步应该做什么
2. 通过发送明确指令给执行器 (autoglm-phone) 来控制手机
3. 执行器会返回操作结果，你需要根据结果继续分析
4. 每个商品都需要截图保留证据
5. 直到你确认所有商品真伪后，输出”所有商品鉴定完成“

指令格式要求：
- 所有发送给执行器的手机操作指令必须用 ```...``` 包裹
- 指令必须清晰、具体的操作步骤，指令要尽量简单，包括但不限于：
  - 软件操作：”在闲鱼APP内搜索关键词”、“向下滑动查看更多帖子”
  - 信息收集：”点开一个帖子并搜集信息，保存信息“、“将屏幕停留在帖子上尽量包含更多信息，等待用户截图”
  - 特殊命令：”截图“ (启用本地函数，特殊命令，只能写两个字)
- 不要在 ``` 外的地方包含指令内容
- 不要在非指令内容中包含 ``` 包裹

示例格式：
```
打开闲鱼APP，并在主页向下滑动2次
```

当前任务：
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
                max_tokens=3000
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


class VerificationAgent:
    """
    验证代理 - 合并策略Agent和鉴定Agent

    职责：
    - 任务理解与分解
    - 制定执行计划
    - 协调PhoneAgent执行手机操作
    - 商品真伪鉴定
    - 生成鉴定报告
    """

    def __init__(
        self,
        phone_agent: PhoneAgentWrapper,
        llm_client: Optional[object] = None,
        output_dir: str = "./output"
    ):
        """
        初始化验证代理

        Args:
            phone_agent: 手机操作代理
            llm_client: LLM客户端 (用于分析)
            output_dir: 输出目录
        """
        self.phone_agent = phone_agent
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.session_dir: Optional[Path] = None

        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        self.llm_temperature = 0.7
        self.llm_max_tokens = 2000

    def run(
        self,
        query: str,
        platform: Platform,
        max_products: int = 5
    ) -> VerificationReport:
        """
        运行商品鉴定流程

        Args:
            query: 搜索关键词
            platform: 平台 (xianyu/xiaohongshu)
            max_products: 最大商品数

        Returns:
            鉴定报告
        """
        platform_str = platform.value if isinstance(
            platform, Platform) else platform
        platform_name = "闲鱼" if platform_str == "xianyu" else "小红书"

        self._create_session_dir(platform_str)

        self._log_header(query, platform_name, max_products)

        # Step 1: 搜索二手平台商品
        self._log("\n[Step 1] 在二手平台搜索商品")
        products = self._search_marketplace(query, platform_name, max_products)

        if not products:
            self._log("  未找到任何商品")
            return self._generate_empty_report(query, platform_str)

        # Step 2: 搜索正品参考
        self._log("\n[Step 2] 搜索正品参考信息")
        authentic_ref = self._search_authentic_reference(query)

        # Step 3: 逐个鉴定商品
        self._log(f"\n[Step 3] 开始鉴定 {len(products)} 个商品")
        results = []
        for i, product in enumerate(products, 1):
            self._log(f"\n  [{i}/{len(products)}] 鉴定: {product.title[:30]}...")
            result = self._verify_product(product, authentic_ref, i)
            results.append(result)
            self._log_verification_result(result)

        # Step 4: 生成报告
        self._log("\n[Step 4] 生成鉴定报告")
        report = self._generate_report(query, platform_str, results)
        self._save_report(report)

        return report

    def _create_session_dir(self, platform: str):
        """创建会话目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / \
            f"{platform}_verification_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "screenshots").mkdir(exist_ok=True)

    def _search_marketplace(
        self,
        query: str,
        platform_name: str,
        max_products: int
    ) -> List[ProductInfo]:
        """
        在二手平台搜索商品

        Args:
            query: 搜索关键词
            platform_name: 平台名称
            max_products: 最大商品数

        Returns:
            商品列表
        """
        instruction = f"在{platform_name}APP搜索'{query}'，浏览前{max_products}个商品，记录每个商品的标题、价格、卖家信息"

        self._log(f"  -> 指令: {instruction[:50]}...")
        result = self.phone_agent.execute(instruction)
        print(result)
        self._log(f"  <- 结果: {result[:100]}...")

        products = self._parse_products_from_result(result, platform_name)

        if not products:
            products = [
                ProductInfo(
                    title=f"{query} - 商品{i+1}",
                    price=0,
                    platform=platform_name,
                    description=result[:200] if result else "",
                )
                for i in range(min(max_products, 3))
            ]

        self._log(f"  找到 {len(products)} 个商品")
        return products[:max_products]

    def _parse_products_from_result(
        self,
        result: str,
        platform: str
    ) -> List[ProductInfo]:
        """解析搜索结果中的商品信息"""
        products = []

        try:
            data = json.loads(result)
            if isinstance(data, dict) and "products" in data:
                for p in data["products"]:
                    products.append(ProductInfo(
                        title=p.get("title", ""),
                        price=float(p.get("price", 0)),
                        platform=platform,
                        description=p.get("description", ""),
                        seller=p.get("seller", ""),
                    ))
        except (json.JSONDecodeError, TypeError):
            pass

        return products

    def _search_authentic_reference(self, query: str) -> dict:
        """
        搜索正品参考信息

        Args:
            query: 搜索关键词

        Returns:
            正品信息字典
        """
        instruction = f"在得到APP搜索'{query}'，记录商品的官方名称、价格、关键特征"

        self._log(f"  -> 指令: {instruction[:50]}...")
        result = self.phone_agent.execute(instruction)
        self._log(f"  <- 结果: {result[:100]}...")

        price_match = re.search(r'[¥￥](\d+\.?\d*)', result)
        price = float(price_match.group(1)) if price_match else 0

        return {
            "title": f"官方正品 - {query}",
            "price": price,
            "description": result[:200] if result else "",
            "seller": "官方渠道"
        }

    def _verify_product(
        self,
        product: ProductInfo,
        authentic_ref: dict,
        index: int
    ) -> VerificationResult:
        """
        鉴定单个商品

        Args:
            product: 商品信息
            authentic_ref: 正品参考
            index: 商品索引

        Returns:
            鉴定结果
        """
        screenshot_path = self._capture_product_screenshot(index)
        product.screenshot_path = screenshot_path

        if self.llm_client:
            return self._analyze_with_llm(product, authentic_ref)
        else:
            return self._basic_analysis(product, authentic_ref)

    def _capture_product_screenshot(self, index: int) -> Optional[str]:
        """获取商品截图"""
        try:
            instruction = f"截图当前商品页面"
            self.phone_agent.execute(instruction)

            screenshot_path = self.session_dir / \
                "screenshots" / f"product_{index}.png"
            return str(screenshot_path)
        except Exception:
            return None

    def _analyze_with_llm(
        self,
        product: ProductInfo,
        authentic_ref: dict
    ) -> VerificationResult:
        """使用LLM进行分析"""
        prompt = f"""
请对比以下二手平台商品与正品信息，进行真伪鉴定:

【二手商品】
标题: {product.title}
价格: ¥{product.price}
描述: {product.description or '未提供'}
卖家: {product.seller or '未提供'}

【正品参考】
标题: {authentic_ref.get('title', '')}
价格: ¥{authentic_ref.get('price', 0)}
描述: {authentic_ref.get('description', '')}

请分析:
1. 价格是否合理
2. 描述是否一致
3. 风险指标
4. 购买建议

请用JSON格式返回:
{{
    "is_authentic": true/false/null,
    "confidence_score": 0.0-1.0,
    "analysis_summary": "分析总结",
    "risk_indicators": ["风险1", "风险2"],
    "recommendations": ["建议1", "建议2"]
}}
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是专业的商品鉴定专家，请客观分析商品真伪。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens
            )

            content = response.choices[0].message.content.strip()
            if content.startswith('```'):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)

            data = json.loads(content)

            return VerificationResult(
                product=product,
                is_authentic=data.get("is_authentic"),
                confidence_score=float(data.get("confidence_score", 0.5)),
                analysis_summary=data.get("analysis_summary", ""),
                risk_indicators=data.get("risk_indicators", []),
                recommendations=data.get("recommendations", [])
            )

        except Exception as e:
            return self._basic_analysis(product, authentic_ref, f"LLM分析失败: {e}")

    def _basic_analysis(
        self,
        product: ProductInfo,
        authentic_ref: dict,
        note: str = ""
    ) -> VerificationResult:
        """基础分析（无LLM时使用）"""
        risk_indicators = []
        recommendations = []

        authentic_price = authentic_ref.get("price", 0)
        if authentic_price > 0 and product.price < authentic_price * 0.5:
            risk_indicators.append("价格过低，可能存在风险")
            recommendations.append("建议核实商品来源")

        if not product.description or len(product.description) < 20:
            risk_indicators.append("描述信息不足")
            recommendations.append("建议要求卖家提供更多详情")

        is_authentic = None if risk_indicators else True
        confidence_score = 0.5 if risk_indicators else 0.7

        summary = note if note else "基础分析完成，建议结合实际情况判断"

        return VerificationResult(
            product=product,
            is_authentic=is_authentic,
            confidence_score=confidence_score,
            analysis_summary=summary,
            risk_indicators=risk_indicators,
            recommendations=recommendations
        )

    def _generate_report(
        self,
        query: str,
        platform: str,
        results: List[VerificationResult]
    ) -> VerificationReport:
        """生成鉴定报告"""
        authentic_count = sum(1 for r in results if r.is_authentic is True)
        suspicious_count = sum(1 for r in results if r.is_authentic is False)
        uncertain_count = sum(1 for r in results if r.is_authentic is None)

        total = len(results)
        authenticity_rate = (authentic_count / total * 100) if total > 0 else 0

        overall_recs = []
        if suspicious_count > total / 2:
            overall_recs.append("该搜索结果中可疑商品较多，请谨慎购买")
        if uncertain_count > 0:
            overall_recs.append("部分商品无法确定真伪，建议进一步核实")

        all_risks = []
        for r in results:
            all_risks.extend(r.risk_indicators)
        risk_counts = {}
        for risk in all_risks:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        for risk, count in sorted(risk_counts.items(), key=lambda x: -x[1])[:3]:
            overall_recs.append(f"常见风险({count}个商品): {risk}")

        return VerificationReport(
            query=query,
            platform=platform,
            timestamp=datetime.now().isoformat(),
            products_analyzed=total,
            authentic_count=authentic_count,
            suspicious_count=suspicious_count,
            uncertain_count=uncertain_count,
            authenticity_rate=round(authenticity_rate, 2),
            results=results,
            overall_recommendations=overall_recs,
            session_dir=str(self.session_dir)
        )

    def _generate_empty_report(self, query: str, platform: str) -> VerificationReport:
        """生成空报告"""
        return VerificationReport(
            query=query,
            platform=platform,
            timestamp=datetime.now().isoformat(),
            products_analyzed=0,
            authentic_count=0,
            suspicious_count=0,
            uncertain_count=0,
            authenticity_rate=0,
            results=[],
            overall_recommendations=["未找到商品，请尝试其他关键词"],
            session_dir=str(self.session_dir) if self.session_dir else ""
        )

    def _save_report(self, report: VerificationReport):
        """保存报告到文件"""
        if not self.session_dir:
            return

        report_dict = {
            "report_info": {
                "query": report.query,
                "platform": report.platform,
                "timestamp": report.timestamp,
                "session_dir": report.session_dir
            },
            "summary": {
                "total_products_analyzed": report.products_analyzed,
                "authentic_count": report.authentic_count,
                "suspicious_count": report.suspicious_count,
                "uncertain_count": report.uncertain_count,
                "authenticity_rate": report.authenticity_rate
            },
            "detailed_results": [
                {
                    "product": {
                        "title": r.product.title,
                        "price": r.product.price,
                        "platform": r.product.platform,
                        "seller": r.product.seller
                    },
                    "verification": {
                        "is_authentic": r.is_authentic,
                        "confidence_score": r.confidence_score,
                        "analysis_summary": r.analysis_summary,
                        "risk_indicators": r.risk_indicators,
                        "recommendations": r.recommendations
                    }
                }
                for r in report.results
            ],
            "overall_recommendations": report.overall_recommendations
        }

        report_file = self.session_dir / "verification_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        self._log(f"  报告已保存: {report_file}")

    def _log_header(self, query: str, platform: str, max_products: int):
        """输出头部信息"""
        print("\n" + "=" * 60)
        print("  Product Verification Agent - 商品鉴定助手")
        print("  Architecture: VerificationAgent + PhoneAgent")
        print("=" * 60)
        print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  搜索关键词: {query}")
        print(f"  目标平台: {platform}")
        print(f"  最大商品数: {max_products}")
        print("=" * 60)

    def _log_verification_result(self, result: VerificationResult):
        """输出鉴定结果"""
        if result.is_authentic is True:
            status = "正品"
        elif result.is_authentic is False:
            status = "可疑"
        else:
            status = "待定"

        print(f"    -> 结果: {status} (置信度: {result.confidence_score:.0%})")
        if result.risk_indicators:
            print(f"    -> 风险: {', '.join(result.risk_indicators[:2])}")

    def _log(self, message: str):
        """日志输出"""
        print(message)


# =============================================================================
# Main Functions
# =============================================================================

def create_llm_client():
    """创建LLM客户端"""
    if not OpenAI:
        print("[Warning] openai package not installed")
        return None

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("[Warning] LLM_API_KEY not configured")
        return None

    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

    try:
        return OpenAI(base_url=base_url, api_key=api_key)
    except Exception as e:
        print(f"[Warning] Failed to create LLM client: {e}")
        return None


def run_verification(
    query: str,
    platform: str = "xianyu",
    max_products: int = 5,
    env_file: str = ".env"
):
    """
    运行商品鉴定

    Args:
        query: 搜索关键词
        platform: 平台名称
        max_products: 最大商品数
        env_file: 环境变量文件
    """
    load_dotenv(env_file)

    platform_enum = Platform.XIANYU if platform == "xianyu" else Platform.XIAOHONGSHU

    phone_agent = PhoneAgentWrapper()
    llm_client = create_llm_client()
    output_dir = os.getenv("OUTPUT_DIR", "./output")

    agent = VerificationAgent(
        phone_agent=phone_agent,
        llm_client=llm_client,
        output_dir=output_dir
    )

    report = agent.run(
        query=query,
        platform=platform_enum,
        max_products=max_products
    )

    print("\n" + "=" * 60)
    print("  鉴定结果摘要")
    print("=" * 60)
    print(f"  总分析商品数: {report.products_analyzed}")
    print(f"  正品: {report.authentic_count}")
    print(f"  可疑: {report.suspicious_count}")
    print(f"  待定: {report.uncertain_count}")
    print(f"  正品率: {report.authenticity_rate}%")
    print("\n  建议:")
    for rec in report.overall_recommendations:
        print(f"    - {rec}")
    print(f"\n  报告目录: {report.session_dir}")
    print("=" * 60)

    return report


def interactive_mode():
    """交互模式"""
    load_dotenv()

    print("\n" + "=" * 60)
    print("  商品鉴定助手 - 交互模式")
    print("  输入 'quit' 退出")
    print("=" * 60)

    phone_agent = PhoneAgentWrapper()
    llm_client = create_llm_client()
    output_dir = os.getenv("OUTPUT_DIR", "./output")

    agent = VerificationAgent(
        phone_agent=phone_agent,
        llm_client=llm_client,
        output_dir=output_dir
    )

    while True:
        print("\n" + "-" * 40)
        query = input("请输入搜索关键词: ").strip()

        if not query:
            print("请输入有效的搜索词")
            continue

        if query.lower() == "quit":
            print("再见!")
            break

        print("\n选择平台:")
        print("  1. 闲鱼")
        print("  2. 小红书")
        choice = input("请选择 (1/2): ").strip()
        platform = Platform.XIANYU if choice == "1" else Platform.XIAOHONGSHU

        max_products = int(os.getenv("MAX_PRODUCTS", "5"))

        try:
            report = agent.run(query, platform, max_products)
            print(f"\n完成! 正品率: {report.authenticity_rate}%")
        except Exception as e:
            print(f"\n错误: {e}")


def get_default_from_env(env_file: str, key: str, default):
    """从.env文件读取默认配置值"""
    try:
        load_dotenv(env_file, override=True)
        value = os.getenv(key)
        if value is not None:
            if key == "MAX_PRODUCTS":
                return int(value)
            elif key == "INTERACTIVE_MODE":
                return value.lower() in ("true", "1", "yes")
            elif key == "SEARCH_QUERY":
                return value if value else default
        return default
    except Exception:
        return default


def parse_args():
    """解析命令行参数"""
    env_file = ".env"

    default_query = get_default_from_env(env_file, "SEARCH_QUERY", "")
    default_max_products = get_default_from_env(env_file, "MAX_PRODUCTS", 5)
    default_interactive = get_default_from_env(
        env_file, "INTERACTIVE_MODE", False)

    parser = argparse.ArgumentParser(
        description="商品鉴定助手 - 对比二手平台商品与正品",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --query "iPhone 15" --platform xianyu
    python main.py --query "Switch游戏机" --platform xiaohongshu -m 3
    python main.py --interactive
        """
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        default=default_query,
        help="搜索关键词"
    )

    parser.add_argument(
        "--platform", "-p",
        type=str,
        default="xianyu",
        choices=["xianyu", "xiaohongshu"],
        help="平台 (默认: xianyu)"
    )

    parser.add_argument(
        "--max-products", "-m",
        type=int,
        default=default_max_products,
        help=f"最大商品数 (默认: {default_max_products})"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=default_interactive,
        help="交互模式"
    )

    parser.add_argument(
        "--env-file", "-e",
        type=str,
        default=env_file,
        help="环境变量文件 (默认: .env)"
    )

    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    assert isinstance(
        args.max_products, int) and args.max_products == 1, f"最大商品数{args.max_products}"

    if args.interactive:
        interactive_mode()
    elif args.query:
        run_verification(
            query=args.query,
            platform=args.platform,
            max_products=args.max_products,
            env_file=args.env_file
        )
    else:
        load_dotenv(args.env_file)

        query = "薛兆丰漫画经济学"
        # query = input("请输入搜索关键词: ").strip()
        if not query:
            print("未输入有效搜索词")
            sys.exit(1)

        print("\n选择平台:")
        print("  1. 闲鱼")
        print("  2. 小红书")
        choice = input("请选择 (1/2): ").strip()
        platform = "xianyu" if choice == "1" else "xiaohongshu"

        run_verification(
            query=query,
            platform=platform,
            max_products=args.max_products,
            env_file=args.env_file
        )


def main_test():
    """
    测试 VerificationAgent_test 双模型协作鉴定系统

    调用 VerificationAgent_test 类，验证总指挥(deepseek) + 执行器(autoglm-phone)
    的双模型协作功能是否正常工作。
    """
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

    agent = VerificationAgent_test(
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
    # main()
    main_test()
