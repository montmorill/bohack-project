"""
Agent1: 提示词生成大模型
功能：根据用户输入的任务需求，为操控模拟器的Agent2生成精炼准确的步骤提示词
"""

from openai import OpenAI
from dataclasses import dataclass
from typing import Any


@dataclass
class PromptGeneratorConfig:
    """提示词生成器配置 - 使用DeepSeek API"""
    base_url: str = "https://api.deepseek.com"
    api_key: str = "sk-261f91400d7c41fd989c807332fbc64d"  # 请替换为你的DeepSeek API Key
    model_name: str = "deepseek-chat"
    max_tokens: int = 2000
    temperature: float = 0.3


# Agent1的系统提示词：指导如何生成Agent2的操作提示词
PROMPT_GENERATOR_SYSTEM = """
你是一个专业的任务分解与提示词生成专家(Agent1)。你的职责是根据用户的任务需求，为操控手机模拟器的执行Agent(Agent2)生成清晰、精炼、可执行的步骤提示词。

## 你的输出要求
1. 分析用户任务的核心目标
2. 将任务分解为具体可执行的步骤
3. 生成适合Agent2执行的系统提示词

## Agent2的能力说明
Agent2是一个视觉语言模型，能够：
- 查看手机屏幕截图
- 执行点击、滑动、输入等操作
- 启动应用、返回、等待等系统操作

## Agent2可用的操作指令
- do(action="Launch", app="xxx") - 启动应用
- do(action="Tap", element=[x,y]) - 点击坐标(0-999范围)
- do(action="Type", text="xxx") - 输入文本
- do(action="Swipe", start=[x1,y1], end=[x2,y2]) - 滑动
- do(action="Back") - 返回上一页
- do(action="Wait", duration="x seconds") - 等待
- do(action="Note", message="xxx") - 记录信息
- finish(message="xxx") - 完成任务

## 生成提示词的原则
1. 步骤清晰：每个步骤目标明确，无歧义
2. 顺序合理：按逻辑顺序排列，考虑依赖关系
3. 容错处理：包含异常情况的处理指引
4. 简洁精炼：去除冗余，保留核心指令
5. 可验证性：每步都有明确的完成标准

## 输出格式
请直接输出为Agent2设计的系统提示词，格式如下：

```
## 任务目标
[一句话描述核心目标]

## 执行步骤
1. [步骤1描述]
2. [步骤2描述]
...

## 关键判断标准
[列出需要Agent2判断的关键点]

## 异常处理
[列出可能的异常及处理方式]

## 完成标准
[明确任务完成的判定条件]
```
"""

# Agent2的基础系统提示词模板
AGENT2_BASE_PROMPT = """
你是一个智能手机操控Agent，根据屏幕截图和任务指引执行操作。

## 输出格式
<think>简短分析当前状态和下一步操作</think>
<answer>具体操作指令</answer>

## 可用操作
- do(action="Launch", app="xxx") - 启动应用
- do(action="Tap", element=[x,y]) - 点击坐标(0-999)
- do(action="Type", text="xxx") - 输入文本
- do(action="Swipe", start=[x1,y1], end=[x2,y2]) - 滑动
- do(action="Back") - 返回
- do(action="Wait", duration="x seconds") - 等待
- do(action="Note", message="xxx") - 记录信息
- finish(message="xxx") - 完成任务

## 执行规则
1. 操作前确认当前app是否正确，否则先Launch
2. 进入无关页面时执行Back返回
3. 页面未加载完成时Wait等待
4. 找不到目标时尝试Swipe滑动查找
5. 操作不生效时适当调整位置重试

{task_prompt}
"""


class PromptGenerator:
    """
    提示词生成器(Agent1)
    调用大模型根据用户需求生成Agent2的操作提示词
    """
    
    def __init__(self, config: PromptGeneratorConfig | None = None):
        self.config = config or PromptGeneratorConfig()
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
    
    def generate_prompt(self, user_task: str, platform: str = "", extra_context: str = "") -> str:
        """
        根据用户任务生成Agent2的系统提示词
        
        Args:
            user_task: 用户的任务描述
            platform: 目标平台(可选，如xianyu/xiaohongshu)
            extra_context: 额外上下文信息
        
        Returns:
            为Agent2生成的完整系统提示词
        """
        # 构建用户消息
        user_message = f"请为以下任务生成Agent2的操作提示词：\n\n任务描述：{user_task}"
        
        if platform:
            user_message += f"\n目标平台：{platform}"
        
        if extra_context:
            user_message += f"\n补充信息：{extra_context}"
        
        # 调用大模型生成提示词
        messages = [
            {"role": "system", "content": PROMPT_GENERATOR_SYSTEM},
            {"role": "user", "content": user_message}
        ]
        
        try:
            print("\n" + "=" * 60)
            print("🤖 Agent1 (DeepSeek) 正在生成提示词...")
            print("=" * 60)
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            task_prompt = response.choices[0].message.content
            
            # 输出生成的提示词
            print("\n📝 DeepSeek 生成的任务提示词:")
            print("-" * 60)
            print(task_prompt)
            print("-" * 60)
            print("✅ 提示词生成完成，开始执行任务...\n")
            
            # 将生成的任务提示词嵌入Agent2基础模板
            full_prompt = AGENT2_BASE_PROMPT.format(task_prompt=task_prompt)
            return full_prompt
            
        except Exception as e:
            print(f"提示词生成失败: {e}")
            # 返回基础模板作为降级方案
            return AGENT2_BASE_PROMPT.format(task_prompt=f"## 任务目标\n{user_task}")
    
    def generate_prompt_sync(self, user_task: str, platform: str = "", extra_context: str = "") -> str:
        """同步版本的提示词生成"""
        return self.generate_prompt(user_task, platform, extra_context)


def get_prompt_for_task(
    task: str,
    platform: str = "",
    extra_context: str = "",
    config: PromptGeneratorConfig | None = None
) -> str:
    """
    便捷函数：根据任务获取Agent2的系统提示词
    
    Args:
        task: 任务描述
        platform: 目标平台
        extra_context: 额外上下文信息
        config: 生成器配置
    
    Returns:
        Agent2的系统提示词
    """
    generator = PromptGenerator(config)
    return generator.generate_prompt(task, platform, extra_context)


# 保持向后兼容的接口
def get_piracy_detection_prompt(platform: str = "xianyu") -> str:
    """
    兼容旧接口：获取盗版检测提示词
    现在会调用大模型动态生成
    """
    task = "在目标平台搜索并检测疑似盗版'得到'App内容的商品，识别低价打包销售、网盘分发等盗版特征，记录证据并进行举报"
    
    platform_context = ""
    if platform == "xianyu":
        platform_context = "闲鱼二手交易平台，搜索栏在顶部，商品采用瀑布流布局"
    elif platform == "xiaohongshu":
        platform_context = "小红书平台，注意筛选商品标签而非普通笔记"
    
    return get_prompt_for_task(task, platform, extra_context=platform_context)


# 保持向后兼容
PIRACY_DETECTION_SYSTEM_PROMPT = AGENT2_BASE_PROMPT.format(
    task_prompt="## 任务目标\n检测并举报盗版内容"
)
