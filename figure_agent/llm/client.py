import os
from typing import List, Dict, Optional, Generator, Any

import httpx
from openai import OpenAI
from openai import APIConnectionError, AuthenticationError, BadRequestError, RateLimitError


class QwenClient:
    """
    阿里云百炼 DashScope 兼容 OpenAI SDK 的 Qwen 客户端。

    重点：
    1. 默认禁用系统代理 trust_env=False，避免 http_proxy/https_proxy 干扰。
    2. 支持非流式 chat。
    3. 支持流式 stream_chat。
    4. 支持 enable_thinking。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-max",
        enable_thinking: bool = True,
        timeout: float = 60.0,
        max_retries: int = 2,
        use_env_proxy: bool = False,
    ):
        self.api_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY".lower())
        )

        if not self.api_key:
            raise ValueError(
                "未检测到 API Key。请设置环境变量 DASHSCOPE_API_KEY，"
                "或者初始化 QwenClient(api_key='sk-xxx')。"
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enable_thinking = enable_thinking

        # 关键修复点：
        # use_env_proxy=False 时，httpx 不读取系统 http_proxy / https_proxy / SSL_CERT_FILE 等环境变量。
        http_client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            trust_env=use_env_proxy,
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = self._build_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "extra_body": {
                "enable_thinking": self.enable_thinking,
            },
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            completion = self.client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            return content or ""

        except AuthenticationError as exc:
            raise RuntimeError(
                "认证失败：请检查 DASHSCOPE_API_KEY 是否正确，是否有百炼模型调用权限。"
            ) from exc

        except BadRequestError as exc:
            raise RuntimeError(
                f"请求参数错误：请检查 model='{self.model}' 是否可用，"
                f"extra_body 是否被当前模型支持。原始错误：{exc}"
            ) from exc

        except RateLimitError as exc:
            raise RuntimeError(
                "触发限流或额度不足：请检查百炼控制台额度、并发限制或计费状态。"
            ) from exc

        except APIConnectionError as exc:
            raise RuntimeError(
                "网络连接失败：通常是代理、TLS、公司网络、防火墙或 base_url 访问异常。\n"
                "建议：\n"
                "1. 先关闭系统代理或设置 use_env_proxy=False。\n"
                "2. 在 PowerShell 执行：curl https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                "3. 检查环境变量 HTTP_PROXY、HTTPS_PROXY 是否存在。\n"
                "4. 如果必须使用代理，把代理配置到 httpx.Client，而不是 base_url。\n"
                f"原始错误：{exc}"
            ) from exc

    def stream_chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        messages = self._build_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "extra_body": {
                "enable_thinking": self.enable_thinking,
            },
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            stream = self.client.chat.completions.create(**kwargs)

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta and delta.content:
                    yield delta.content

        except AuthenticationError as exc:
            raise RuntimeError(
                "认证失败：请检查 DASHSCOPE_API_KEY 是否正确。"
            ) from exc

        except APIConnectionError as exc:
            raise RuntimeError(
                "流式请求网络连接失败：大概率是代理或 TLS 握手问题。"
            ) from exc

    def chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
        max_tokens: Optional[int] = None,
    ):
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "extra_body": {
                "enable_thinking": self.enable_thinking,
            },
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _build_messages(
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        return messages

    def close(self) -> None:
        self.client.close()


if __name__ == "__main__":
    client = QwenClient(
        model="qwen3-max",
        enable_thinking=True,
        use_env_proxy=False,  # 默认不走系统代理
    )

    print("非流式测试：")
    print(client.chat("你是谁"))

    print("\n流式测试：")
    for text in client.stream_chat("用一句话介绍你自己"):
        print(text, end="", flush=True)

    client.close()