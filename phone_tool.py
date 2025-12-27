#!/usr/bin/env python3
"""
Phone Tool - 手机操作工具封装

将 main_autoglm.py 的功能封装为可被其他大模型调用的工具接口。

支持的调用方式:
1. 直接函数调用
2. 类实例调用
3. OpenAI Function Calling 格式
4. Tool Schema (用于 Claude, GPT-4 等)

Usage:
    # 方式1: 直接函数调用
    from phone_tool import execute_phone_action
    result = execute_phone_action("打开微信，搜索张三")
    
    # 方式2: 类实例调用
    from phone_tool import PhoneTool
    tool = PhoneTool()
    result = tool.execute("在闲鱼搜索iPhone 15")
    
    # 方式3: 获取工具描述供大模型使用
    from phone_tool import get_tool_schema, get_openai_function_schema
    schema = get_tool_schema()  # 通用schema
    openai_schema = get_openai_function_schema()  # OpenAI格式

Environment Variables:
    PHONE_AGENT_BASE_URL: AutoGLM API base URL
    PHONE_AGENT_MODEL: AutoGLM model name  
    PHONE_AGENT_API_KEY: AutoGLM API key
    PHONE_AGENT_DEVICE_ID: ADB device ID
    PHONE_AGENT_MAX_STEPS: Maximum steps per task
"""

import os
import json
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

from dotenv import load_dotenv


# =============================================================================
# Data Classes
# =============================================================================

class ActionStatus(Enum):
    """操作状态"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ActionResult:
    """操作结果"""
    status: ActionStatus
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    screenshot_path: Optional[str] = None
    steps_executed: int = 0
    execution_time: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        result = asdict(self)
        result["status"] = self.status.value
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class PhoneToolConfig:
    """Phone Tool 配置"""
    base_url: str = "http://localhost:8000/v1"
    model: str = "autoglm-phone-9b"
    api_key: str = "EMPTY"
    device_id: Optional[str] = None
    max_steps: int = 100
    lang: str = "cn"
    verbose: bool = True
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "PhoneToolConfig":
        """从环境变量加载配置"""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        return cls(
            base_url=os.getenv("PHONE_AGENT_BASE_URL", "http://localhost:8000/v1"),
            model=os.getenv("PHONE_AGENT_MODEL", "autoglm-phone-9b"),
            api_key=os.getenv("PHONE_AGENT_API_KEY", "EMPTY"),
            device_id=os.getenv("PHONE_AGENT_DEVICE_ID"),
            max_steps=int(os.getenv("PHONE_AGENT_MAX_STEPS", "100")),
            lang=os.getenv("PHONE_AGENT_LANG", "cn"),
            verbose=os.getenv("PHONE_AGENT_VERBOSE", "true").lower() == "true"
        )


# =============================================================================
# Phone Tool Class
# =============================================================================

class PhoneTool:
    """
    手机操作工具 - 封装 AutoGLM 的手机控制功能
    
    可被其他大模型作为工具调用，执行手机上的各种操作。
    
    Example:
        tool = PhoneTool()
        result = tool.execute("打开微信，给张三发消息说你好")
        print(result.message)
    """
    
    # 工具名称和描述
    TOOL_NAME = "phone_control"
    TOOL_DESCRIPTION = """
    手机控制工具 - 通过自然语言指令控制Android手机执行操作。
    
    支持的操作类型:
    - 打开/关闭应用程序
    - 搜索商品、内容
    - 点击、滑动、输入文字
    - 截图保存
    - 浏览页面、获取信息
    
    使用示例:
    - "打开闲鱼APP，搜索iPhone 15"
    - "在微信中给张三发消息"
    - "打开得到APP，搜索薛兆丰的课程"
    - "截图当前屏幕"
    - "向下滑动页面"
    """
    
    def __init__(self, config: Optional[PhoneToolConfig] = None):
        """
        初始化手机工具
        
        Args:
            config: 配置对象，为None时从环境变量加载
        """
        self.config = config or PhoneToolConfig.from_env()
        self._agent = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """确保Agent已初始化"""
        if self._initialized:
            return
        
        try:
            from phone_agent import PhoneAgent
            from phone_agent.agent import AgentConfig
            from phone_agent.device_factory import set_device_type
            from phone_agent.model import ModelConfig
            
            set_device_type("adb")
            
            model_config = ModelConfig(
                model_name=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                lang=self.config.lang,
            )
            
            agent_config = AgentConfig(
                max_steps=self.config.max_steps,
                device_id=self.config.device_id,
                verbose=self.config.verbose,
                lang=self.config.lang,
            )
            
            self._agent = PhoneAgent(
                model_config=model_config,
                agent_config=agent_config,
            )
            self._initialized = True
            
        except ImportError as e:
            raise RuntimeError(f"Failed to import phone_agent: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PhoneAgent: {e}")
    
    def execute(self, instruction: str) -> ActionResult:
        """
        执行手机操作指令
        
        Args:
            instruction: 自然语言指令，描述要执行的操作
            
        Returns:
            ActionResult: 操作结果
            
        Example:
            result = tool.execute("打开闲鱼搜索iPhone")
            if result.status == ActionStatus.SUCCESS:
                print(result.message)
        """
        if not instruction or not instruction.strip():
            return ActionResult(
                status=ActionStatus.FAILED,
                message="指令不能为空",
                error="Empty instruction"
            )
        
        start_time = datetime.now()
        
        try:
            self._ensure_initialized()
            
            result = self._agent.run(instruction)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=str(result) if result else "操作完成",
                steps_executed=self._agent.step_count,
                execution_time=execution_time,
                data={"raw_result": str(result)}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ActionResult(
                status=ActionStatus.FAILED,
                message=f"操作失败: {str(e)}",
                error=str(e),
                execution_time=execution_time
            )
    
    def reset(self):
        """重置Agent状态，用于开始新任务"""
        if self._agent:
            self._agent.reset()
    
    def __call__(self, instruction: str) -> ActionResult:
        """支持直接调用实例"""
        return self.execute(instruction)
    
    @classmethod
    def get_tool_schema(cls) -> Dict[str, Any]:
        """
        获取工具Schema描述
        
        用于向大模型描述此工具的功能和参数格式。
        兼容多种大模型的tool/function格式。
        
        Returns:
            工具描述的字典格式
        """
        return {
            "name": cls.TOOL_NAME,
            "description": cls.TOOL_DESCRIPTION.strip(),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "要执行的手机操作指令，使用自然语言描述。例如：'打开闲鱼搜索iPhone 15'、'在微信给张三发消息'、'截图当前屏幕'"
                    }
                },
                "required": ["instruction"]
            }
        }
    
    @classmethod
    def get_openai_function_schema(cls) -> Dict[str, Any]:
        """
        获取OpenAI Function Calling格式的Schema
        
        Returns:
            OpenAI functions格式的字典
        """
        return {
            "type": "function",
            "function": cls.get_tool_schema()
        }
    
    @classmethod
    def get_anthropic_tool_schema(cls) -> Dict[str, Any]:
        """
        获取Anthropic Claude Tool Use格式的Schema
        
        Returns:
            Claude tools格式的字典
        """
        return {
            "name": cls.TOOL_NAME,
            "description": cls.TOOL_DESCRIPTION.strip(),
            "input_schema": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "要执行的手机操作指令，使用自然语言描述"
                    }
                },
                "required": ["instruction"]
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

# 全局工具实例（懒加载）
_global_tool: Optional[PhoneTool] = None


def _get_global_tool() -> PhoneTool:
    """获取全局工具实例"""
    global _global_tool
    if _global_tool is None:
        _global_tool = PhoneTool()
    return _global_tool


def execute_phone_action(instruction: str) -> ActionResult:
    """
    执行手机操作（便捷函数）
    
    Args:
        instruction: 自然语言指令
        
    Returns:
        ActionResult: 操作结果
        
    Example:
        from phone_tool import execute_phone_action
        result = execute_phone_action("打开微信")
        print(result.message)
    """
    return _get_global_tool().execute(instruction)


def execute_phone_action_simple(instruction: str) -> str:
    """
    执行手机操作并返回简单字符串结果
    
    适合作为其他大模型的工具函数使用。
    
    Args:
        instruction: 自然语言指令
        
    Returns:
        str: 操作结果描述
        
    Example:
        result = execute_phone_action_simple("打开闲鱼搜索iPhone")
        # 返回: "操作成功: 已打开闲鱼并搜索iPhone"
    """
    result = execute_phone_action(instruction)
    if result.status == ActionStatus.SUCCESS:
        return f"操作成功: {result.message}"
    else:
        return f"操作失败: {result.message}"


def get_tool_schema() -> Dict[str, Any]:
    """获取工具Schema（通用格式）"""
    return PhoneTool.get_tool_schema()


def get_openai_function_schema() -> Dict[str, Any]:
    """获取OpenAI Function格式Schema"""
    return PhoneTool.get_openai_function_schema()


def get_anthropic_tool_schema() -> Dict[str, Any]:
    """获取Anthropic Claude格式Schema"""
    return PhoneTool.get_anthropic_tool_schema()


# =============================================================================
# Tool Handler for LLM Integration
# =============================================================================

class PhoneToolHandler:
    """
    手机工具处理器 - 用于集成到LLM工具调用流程
    
    Example:
        handler = PhoneToolHandler()
        
        # 在LLM返回tool_call时处理
        if tool_call.name == "phone_control":
            result = handler.handle_tool_call(tool_call.arguments)
    """
    
    def __init__(self, config: Optional[PhoneToolConfig] = None):
        self.tool = PhoneTool(config)
    
    def handle_tool_call(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理工具调用
        
        Args:
            arguments: 工具调用参数 {"instruction": "..."}
            
        Returns:
            工具调用结果字典
        """
        instruction = arguments.get("instruction", "")
        result = self.tool.execute(instruction)
        return result.to_dict()
    
    def handle_tool_call_json(self, arguments_json: str) -> str:
        """
        处理JSON格式的工具调用
        
        Args:
            arguments_json: JSON格式的参数字符串
            
        Returns:
            JSON格式的结果字符串
        """
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError:
            arguments = {"instruction": arguments_json}
        
        result = self.handle_tool_call(arguments)
        return json.dumps(result, ensure_ascii=False)
    
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """获取工具列表（OpenAI格式）"""
        return [self.tool.get_openai_function_schema()]


# =============================================================================
# Example Integration Functions
# =============================================================================

def create_openai_tools_config() -> List[Dict[str, Any]]:
    """
    创建OpenAI API调用时的tools配置
    
    Example:
        from openai import OpenAI
        from phone_tool import create_openai_tools_config, PhoneToolHandler
        
        client = OpenAI()
        handler = PhoneToolHandler()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "帮我在闲鱼搜索iPhone 15"}],
            tools=create_openai_tools_config()
        )
        
        # 处理tool_calls
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.function.name == "phone_control":
                result = handler.handle_tool_call_json(tool_call.function.arguments)
    """
    return [get_openai_function_schema()]


def create_anthropic_tools_config() -> List[Dict[str, Any]]:
    """
    创建Anthropic Claude API调用时的tools配置
    
    Example:
        import anthropic
        from phone_tool import create_anthropic_tools_config, PhoneToolHandler
        
        client = anthropic.Anthropic()
        handler = PhoneToolHandler()
        
        response = client.messages.create(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "帮我在闲鱼搜索iPhone 15"}],
            tools=create_anthropic_tools_config()
        )
        
        # 处理tool_use
        for block in response.content:
            if block.type == "tool_use" and block.name == "phone_control":
                result = handler.handle_tool_call(block.input)
    """
    return [get_anthropic_tool_schema()]


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phone Tool - 手机操作工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 执行操作
    python phone_tool.py "打开微信"
    python phone_tool.py "在闲鱼搜索iPhone 15"
    
    # 交互模式
    python phone_tool.py --interactive
    
    # 输出工具Schema
    python phone_tool.py --schema
    python phone_tool.py --schema openai
    python phone_tool.py --schema anthropic
        """
    )
    
    parser.add_argument(
        "instruction",
        nargs="?",
        help="要执行的手机操作指令"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互模式"
    )
    
    parser.add_argument(
        "--schema", "-s",
        nargs="?",
        const="default",
        choices=["default", "openai", "anthropic"],
        help="输出工具Schema"
    )
    
    parser.add_argument(
        "--env-file", "-e",
        type=str,
        default=".env",
        help="环境变量文件"
    )
    
    args = parser.parse_args()
    
    load_dotenv(args.env_file)
    
    # 输出Schema
    if args.schema:
        if args.schema == "openai":
            schema = get_openai_function_schema()
        elif args.schema == "anthropic":
            schema = get_anthropic_tool_schema()
        else:
            schema = get_tool_schema()
        print(json.dumps(schema, ensure_ascii=False, indent=2))
        return
    
    # 交互模式
    if args.interactive:
        print("\n" + "=" * 50)
        print("  Phone Tool - 交互模式")
        print("  输入 'quit' 退出")
        print("=" * 50)
        
        tool = PhoneTool()
        
        while True:
            instruction = input("\n请输入指令: ").strip()
            
            if not instruction:
                continue
            
            if instruction.lower() == "quit":
                print("再见!")
                break
            
            print(f"\n执行: {instruction}")
            result = tool.execute(instruction)
            print(f"状态: {result.status.value}")
            print(f"结果: {result.message}")
            
            if result.error:
                print(f"错误: {result.error}")
            
            tool.reset()
        return
    
    # 执行单个指令
    if args.instruction:
        tool = PhoneTool()
        result = tool.execute(args.instruction)
        print(result.to_json())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
