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


if __name__ == "__main__":
    main()
