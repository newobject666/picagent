# figure_agent/agent/llm_service.py

from typing import Optional

from figure_agent.llm.client import QwenClient


class LLMService:
    """
    LLM 服务单例。

    作用：
    1. 全局只初始化一个 QwenClient
    2. 避免多个模块重复创建 client
    3. 统一管理模型参数
    """

    _client: Optional[QwenClient] = None

    @classmethod
    def get_client(cls) -> QwenClient:
        if cls._client is None:
            cls._client = QwenClient(
                model="qwen3-max",
                enable_thinking=True,
                use_env_proxy=False,
            )
            print("▸ LLMService 初始化完成")
            print("   模型: qwen3-max")

        return cls._client

    @classmethod
    def close(cls) -> None:
        if cls._client is not None:
            cls._client.close()
            cls._client = None