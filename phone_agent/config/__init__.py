"""Configuration module for Phone Agent."""

from phone_agent.config.apps import APP_PACKAGES
from phone_agent.config.i18n import get_message, get_messages
from phone_agent.config.prompts_zh import SYSTEM_PROMPT as SYSTEM_PROMPT_ZH
from phone_agent.config.prompts_piracy import (
    PIRACY_DETECTION_SYSTEM_PROMPT,
    get_piracy_detection_prompt,
)
from phone_agent.config.timing import (
    TIMING_CONFIG,
    ActionTimingConfig,
    ConnectionTimingConfig,
    DeviceTimingConfig,
    TimingConfig,
    get_timing_config,
    update_timing_config,
)


def get_system_prompt(lang: str = "cn") -> str:
    """Get system prompt by language."""
    return SYSTEM_PROMPT_ZH


SYSTEM_PROMPT = SYSTEM_PROMPT_ZH

__all__ = [
    "APP_PACKAGES",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_ZH",
    "PIRACY_DETECTION_SYSTEM_PROMPT",
    "get_system_prompt",
    "get_piracy_detection_prompt",
    "get_messages",
    "get_message",
    "TIMING_CONFIG",
    "TimingConfig",
    "ActionTimingConfig",
    "DeviceTimingConfig",
    "ConnectionTimingConfig",
    "get_timing_config",
    "update_timing_config",
]
