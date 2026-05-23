import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class HookAction:
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


class HookEvent:
    USER_INPUT = "user_input"
    PLAN_CREATED = "plan_created"
    BEFORE_RAG = "before_rag"
    AFTER_RAG = "after_rag"
    BEFORE_GENERATION = "before_generation"
    AFTER_GENERATION = "after_generation"


@dataclass
class HookContext:
    event: str
    user_input: str = ""
    plan: Any = None
    skill_names: List[str] = field(default_factory=list)
    rag_result: Any = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookDecision:
    action: str
    reason: str
    hook_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, hook_name: str, reason: str = "allowed") -> "HookDecision":
        return cls(action=HookAction.ALLOW, reason=reason, hook_name=hook_name)

    @classmethod
    def warn(cls, hook_name: str, reason: str) -> "HookDecision":
        return cls(action=HookAction.WARN, reason=reason, hook_name=hook_name)

    @classmethod
    def block(cls, hook_name: str, reason: str) -> "HookDecision":
        return cls(action=HookAction.BLOCK, reason=reason, hook_name=hook_name)


@dataclass
class HookRunResult:
    allowed: bool
    decisions: List[HookDecision]

    @property
    def blocks(self) -> List[HookDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.action == HookAction.BLOCK
        ]

    @property
    def warnings(self) -> List[HookDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.action == HookAction.WARN
        ]


class AgentHook(Protocol):
    name: str

    def handle(self, context: HookContext) -> HookDecision:
        ...


class HookManager:
    def __init__(self, hooks: Optional[List[AgentHook]] = None):
        self.hooks: List[AgentHook] = hooks or []

    @classmethod
    def with_default_hooks(cls) -> "HookManager":
        return cls(hooks=[SafetyPolicyHook()])

    def register(self, hook: AgentHook) -> None:
        self.hooks.append(hook)

    def run(self, context: HookContext) -> HookRunResult:
        decisions: List[HookDecision] = []

        for hook in self.hooks:
            decision = hook.handle(context)
            decisions.append(decision)

            if decision.action == HookAction.BLOCK:
                break

        return HookRunResult(
            allowed=not any(
                decision.action == HookAction.BLOCK
                for decision in decisions
            ),
            decisions=decisions,
        )


class SafetyPolicyHook:
    name = "safety_policy"

    blocked_input_patterns = (
        r"\brm\s+-rf\s+/",
        r"\bformat\s+[a-z]:",
        r"\bdel\s+/s\s+/q\s+[a-z]:\\",
        r"powershell.+-encodedcommand",
        r"绕过(认证|权限|登录|安全)",
        r"窃取|偷取|盗取|泄露.*(密码|密钥|token|api key|凭证)",
        r"恶意软件|木马|勒索|后门|病毒",
        r"删除.*(系统|硬盘|数据库|所有文件)",
        r"格式化.*(硬盘|磁盘|系统)",
    )
    admin_only_ingestion_patterns = (
        r"crawl_papers",
        r"manage\.py\s+crawl_papers",
        r"(爬取|抓取|采集).*(论文|paper|arxiv).*(写入|导入|保存|入库|mysql|数据库|论文库)",
        r"(写入|导入|保存|入库).*(mysql|数据库|论文库).*(论文|paper|arxiv)",
        r"(自动|帮我|直接).*(爬取|抓取|采集).*(\d+\s*条)?.*(论文|paper|arxiv)",
    )
    sensitive_output_patterns = (
        r"-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bsk-[A-Za-z0-9_\-]{20,}\b",
        r"DASHSCOPE_API_KEY\s*=\s*['\"]?[\w\-]+",
        r"MYSQL_PASSWORD\s*=\s*['\"]?[^'\"]+",
    )

    def handle(self, context: HookContext) -> HookDecision:
        if context.event == HookEvent.USER_INPUT:
            return self._check_user_input(context)

        if context.event == HookEvent.BEFORE_GENERATION:
            return self._check_before_generation(context)

        if context.event == HookEvent.AFTER_GENERATION:
            return self._check_after_generation(context)

        return HookDecision.allow(self.name)

    def _check_user_input(self, context: HookContext) -> HookDecision:
        text = context.user_input.lower()

        for pattern in self.blocked_input_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return HookDecision.block(
                    self.name,
                    "请求包含高风险或破坏性意图，已被安全 Hook 拦截。",
                )

        for pattern in self.admin_only_ingestion_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return HookDecision.block(
                    self.name,
                    (
                        "论文爬取入库属于后台管理员数据维护操作，不能由普通对话触发。"
                        "请通过后台管理命令或受控管理入口执行。"
                    ),
                )

        return HookDecision.allow(self.name)

    def _check_before_generation(self, context: HookContext) -> HookDecision:
        rag_result = context.rag_result

        if rag_result is not None and getattr(rag_result, "status", "") != "PASS":
            return HookDecision.block(
                self.name,
                "RAG 证据门控未通过，禁止进入自由生成。",
            )

        if rag_result is not None:
            context_text = "\n".join(
                message.get("content", "")
                for message in context.messages
            )

            if "证据约束生成规则" not in context_text:
                return HookDecision.block(
                    self.name,
                    "RAG 回答缺少证据约束生成规则，禁止生成。",
                )

        return HookDecision.allow(self.name)

    def _check_after_generation(self, context: HookContext) -> HookDecision:
        for pattern in self.sensitive_output_patterns:
            if re.search(pattern, context.answer, flags=re.IGNORECASE):
                return HookDecision.block(
                    self.name,
                    "回答疑似包含密钥、凭证或私钥，已被安全 Hook 拦截。",
                )

        return HookDecision.allow(self.name)
