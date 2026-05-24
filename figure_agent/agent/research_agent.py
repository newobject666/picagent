# figure_agent/agent/research_agent.py

import json
from dataclasses import dataclass
from typing import List, Dict, Iterator, Optional, Any

from figure_agent.agent.prompt import get_default_system_prompt
from figure_agent.agent.memory import ConversationMemory
from figure_agent.agent.auto_memory import AutoMemoryStore
from figure_agent.agent.skill_loader import SkillLoader
from figure_agent.agent.llm_service import LLMService
from figure_agent.agent.dynamic_planner import (
    AgentStepResult,
    AgentToolSpec,
    DynamicAgentPlanner,
    DynamicPlanStep,
)
from figure_agent.agent.runner import AgentExecutor, AgentPlan
from figure_agent.agent.mcp_tools import MCPToolRegistry
from figure_agent.agent.hooks import HookContext, HookEvent, HookManager, HookRunResult
from figure_agent.rag.paper_retriever import PaperRAGRetriever, RagRetrievalResult
from pathlib import Path
import logging

logger = logging.getLogger("picagent")


@dataclass
class AgentRouteDecision:
    mode: str
    reason: str
    use_agent_chain: bool

class ResearchAgent:
    """
    科研助手总调度 Agent。

    职责：
    1. 管理多轮对话记忆
    2. 根据用户问题加载对应 skill.md
    3. 判断是否需要启用本地论文库 RAG
    4. 将 skill 内容和 RAG 论文摘要临时注入本轮上下文
    5. 调用 LLM 生成回答
    6. 保存用户问题和模型回答
    7. 压缩旧历史对话

    注意：
    - skill.md 不写入 memory
    - RAG 摘要不写入 memory
    - memory 只保存 user 和 assistant 的真实对话
    """

    def __init__(
        self,
        skill_dir: str = "Skill",
        max_recent_rounds: int = 3,
        max_tokens: int = 1500,
    ):
        self.client = LLMService.get_client()

        project_root = Path(__file__).resolve().parents[2]

        skill_path = Path(skill_dir)
        if not skill_path.is_absolute():
            skill_path = project_root / skill_path

        logger.info(f"ResearchAgent project_root: {project_root}")
        logger.info(f"ResearchAgent skill_path: {skill_path}")
        logger.info(f"skill_path exists: {skill_path.exists()}")

        self.memory = ConversationMemory(
            system_prompt=get_default_system_prompt(),
            max_recent_rounds=max_recent_rounds,
        )
        self.hooks = HookManager.with_default_hooks()
        self.long_term_memories: List[str] = []
        self.runtime_documents: List[Dict[str, Any]] = []
        self.auto_memory = AutoMemoryStore(root_dir=project_root / "memory")

        self.skill_loader = SkillLoader(
            skill_dir=str(skill_path),
            cache_enabled=True,
        )
        self.dynamic_planner = DynamicAgentPlanner(self.client)
        self.mcp_tools = MCPToolRegistry.from_env()

        logger.info(f"可用 skills: {self.skill_loader.available_skills()}")

        self.paper_rag = PaperRAGRetriever(
            paper_library_path=str(project_root / "PaperLibrary" / "papers.json"),
            final_top_k=3,
            candidate_top_k=50,
            full_rerank_threshold=200,
            max_summary_chars=500,
            max_total_chars=2500,
            enable_reranker=True,
            reranker_use_fp16=False,
            enable_bge_embedding=True,
            embedding_use_fp16=False,
            enable_faiss_vector_store=True,
            vector_index_path=str(project_root / "models_cache" / "faiss" / "paper_vectors.index"),
            vector_index_type="ivf_flat",
            vector_nlist=16,
            vector_nprobe=4,
            auto_reload_json=False,
        )

        self.executor = AgentExecutor(self)
        self.max_tokens = max_tokens

    def ask_stream(self, user_input: str) -> Iterator[str]:
        """
        流式对话入口。

        本轮上下文结构：

        system prompt
        + 本轮触发的 skill.md
        + 本轮 RAG 命中的 top3 论文摘要
        + 历史摘要 summary
        + 最近 2 轮完整对话
        + 当前用户输入
        """

        user_input = user_input.strip()
        logger.info("ResearchAgent.ask_stream 开始")
        logger.info(f"user_input: {user_input}")
        if not user_input:
            return

        input_hook_result = self.hooks.run(
            HookContext(
                event=HookEvent.USER_INPUT,
                user_input=user_input,
            )
        )
        if not input_hook_result.allowed:
            yield self._build_hook_refusal(input_hook_result)
            return
        
        # 1. 当前用户输入写入 memory
        if self._should_use_lightweight_reply(user_input):
            answer = self._build_lightweight_reply(user_input)
            self.memory.add_user_message(user_input)
            self.memory.add_assistant_message(answer)
            yield answer
            return

        self.memory.add_user_message(user_input)

        # 2. 检测本轮需要哪些 skill，并生成 plan-to-solve
        skill_names = self.skill_loader.detect_skills(user_input)
        use_rag = self._should_use_paper_rag(
            user_input=user_input,
            skill_names=skill_names,
        )
        self._last_route_decision = self._decide_agent_route(
            user_input=user_input,
            skill_names=skill_names,
            use_rag=use_rag,
        )
        logger.info(
            "Agent route decision: mode=%s reason=%s",
            self._last_route_decision.mode,
            self._last_route_decision.reason,
        )
        plan = self._build_plan(
            user_input=user_input,
            skill_names=skill_names,
            use_rag=use_rag,
        )
        plan_hook_result = self.hooks.run(
            HookContext(
                event=HookEvent.PLAN_CREATED,
                user_input=user_input,
                plan=plan,
                skill_names=skill_names,
            )
        )
        if not plan_hook_result.allowed:
            answer = self._build_hook_refusal(plan_hook_result)
            self.memory.add_assistant_message(answer)
            self._record_auto_memory(user_input, answer)
            yield answer
            return
        if self._should_show_agent_trace():
            yield self._format_plan(plan)

        # 3. 按计划构造 skill messages
        # 注意：build_skill_messages 内部会读取对应 skill.md
        execution_trace = self.executor.execute(
            user_input=user_input,
            plan=plan,
            skill_names=skill_names,
        )
        for update in execution_trace.user_updates:
            if self._should_show_agent_trace():
                yield update

        skill_messages = execution_trace.skill_messages
        use_rag = False
        display_skill_names = skill_names if self._should_show_agent_trace() else []

        if display_skill_names:
            yield "▸ 触发技能: " + ", ".join(skill_names) + "\n\n"
        logger.info(f"触发 skill: {skill_names}")
        # 4. 按计划启用本地论文库 RAG，并检验证据结果
        rag_messages: List[Dict[str, str]] = execution_trace.rag_messages
        rag_messages: List[Dict[str, str]] = execution_trace.rag_messages
        rag_result: Optional[RagRetrievalResult] = execution_trace.rag_result

        if use_rag:
            logger.info("开始执行 PaperRAGRetriever")
            before_rag_result = self.hooks.run(
                HookContext(
                    event=HookEvent.BEFORE_RAG,
                    user_input=user_input,
                    plan=plan,
                    skill_names=skill_names,
                )
            )
            if not before_rag_result.allowed:
                answer = self._build_hook_refusal(before_rag_result)
                self.memory.add_assistant_message(answer)
                self._record_auto_memory(user_input, answer)
                yield answer
                return

            rag_result = self.paper_rag.retrieve_with_guardrails(
                query=user_input,
            )
            after_rag_result = self.hooks.run(
                HookContext(
                    event=HookEvent.AFTER_RAG,
                    user_input=user_input,
                    plan=plan,
                    skill_names=skill_names,
                    rag_result=rag_result,
                )
            )
            if not after_rag_result.allowed:
                answer = self._build_hook_refusal(after_rag_result)
                self.memory.add_assistant_message(answer)
                self._record_auto_memory(user_input, answer)
                yield answer
                return
            rag_message = rag_result.context_message

            if rag_message:
                rag_messages.append(rag_message)
                if rag_result.supplemental_queries:
                    yield (
                        "▸ RAG 已启用：初始证据不足，已触发补充检索并融合重排。\n\n"
                    )
                else:
                    yield (
                        "▸ RAG 已启用：已完成意图识别、混合召回、融合重排和证据门控。\n\n"
                    )

            yield self._format_rag_verification(rag_result)

            if rag_result.status == "REFUSE":
                answer = self._build_refusal_answer(rag_result)
                self.memory.add_assistant_message(answer)
                self._record_auto_memory(user_input, answer)
                self.memory.compress_if_needed(self.client)
                yield answer
                return
            if rag_message:
                logger.info("RAG 命中，已构造上下文 message")
            else:
                logger.info("RAG 未命中或未返回上下文")

        # 5. 生成前执行上下文窗口管理：旧对话压缩、大工具结果摘取、必要时 context reset
        if rag_result is not None and rag_result.status == "REFUSE":
            answer = self._build_refusal_answer(rag_result)
            self.memory.add_assistant_message(answer)
            self._record_auto_memory(user_input, answer)
            self.memory.compress_if_needed(self.client)
            yield answer
            return

        self.memory.compress_if_needed(self.client)

        # 6. 获取基础 memory messages
        base_messages = self.memory.get_messages()

        # 7. 生成前二次证据覆盖校验，未覆盖时禁止随机生成
        pre_generation_refusal = self._pre_generation_evidence_check(rag_result)
        if pre_generation_refusal:
            self.memory.add_assistant_message(pre_generation_refusal)
            self._record_auto_memory(user_input, pre_generation_refusal)
            self.memory.compress_if_needed(self.client)
            yield pre_generation_refusal
            return

        # 8. 将 skill + RAG 作为本轮临时上下文注入
        memory_messages = self._build_long_term_memory_messages(user_input)
        trace_messages = self._build_agent_trace_messages(execution_trace.results)
        document_messages = self._build_runtime_document_messages()
        extra_messages = (
            memory_messages
            + document_messages
            + trace_messages
            + skill_messages
            + rag_messages
        )

        messages = self._inject_extra_system_messages(
            messages=base_messages,
            extra_messages=extra_messages,
        )
        before_generation_hook_result = self.hooks.run(
            HookContext(
                event=HookEvent.BEFORE_GENERATION,
                user_input=user_input,
                plan=plan,
                skill_names=skill_names,
                rag_result=rag_result,
                messages=messages,
            )
        )
        if not before_generation_hook_result.allowed:
            answer = self._build_hook_refusal(before_generation_hook_result)
            self.memory.add_assistant_message(answer)
            self._record_auto_memory(user_input, answer)
            self.memory.compress_if_needed(self.client)
            yield answer
            return
        logger.info(f"最终 messages 数量: {len(messages)}")
        logger.info("开始调用 LLM")
        # 7. 调用 LLM
        full_answer = ""

        try:
            stream = self.client.chat_with_messages(
                messages=messages,
                stream=True,
                max_tokens=self.max_tokens,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta and delta.content:
                    full_answer += delta.content
            logger.info(f"LLM 回答完成，回答长度: {len(full_answer)}")
            after_generation_hook_result = self.hooks.run(
                HookContext(
                    event=HookEvent.AFTER_GENERATION,
                    user_input=user_input,
                    plan=plan,
                    skill_names=skill_names,
                    rag_result=rag_result,
                    messages=messages,
                    answer=full_answer,
                )
            )
            if not after_generation_hook_result.allowed:
                full_answer = self._build_hook_refusal(after_generation_hook_result)
            yield full_answer

            # 8. 保存 assistant 回复
            self.memory.add_assistant_message(full_answer)
            self._record_auto_memory(user_input, full_answer)

            # 9. 按 200K 上下文策略管理历史
            self.memory.compress_if_needed(self.client)

        except Exception as exc:
            # 请求失败时删除本轮 user，避免污染上下文
            logger.exception("ResearchAgent 执行失败")
            self.memory.remove_last_user_message()
            yield "\n[错误] " + str(exc) + "\n"
    
    def ask(self, user_input: str) -> str:
        """
        非流式对话入口。
        """
        chunks: List[str] = []

        for chunk in self.ask_stream(user_input):
            chunks.append(chunk)

        return "".join(chunks)

    def clear_memory(self) -> None:
        """
        清空对话记忆。
        """
        self.memory.clear()

    def load_short_term_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        从数据库历史消息恢复当前会话的短期记忆。
        """
        self.memory.load_messages(messages)
        self.memory.compress_if_needed(self.client)

    def set_long_term_memories(self, memories: List[str]) -> None:
        """
        设置回答时要注入的长期记忆。
        """
        self.long_term_memories = [
            memory.strip()
            for memory in memories
            if memory and memory.strip()
        ]

    def set_runtime_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.runtime_documents = [
            document
            for document in documents
            if document.get("text")
        ][:5]

    def clear_runtime_documents(self) -> None:
        self.runtime_documents = []

    @staticmethod
    def _should_show_agent_trace() -> bool:
        return False

    def _decide_agent_route(
        self,
        user_input: str,
        skill_names: List[str],
        use_rag: bool,
    ) -> AgentRouteDecision:
        if getattr(self, "runtime_documents", []):
            return AgentRouteDecision(
                mode="agent_chain",
                reason="本轮包含上传文档，需要执行 document_context 工具",
                use_agent_chain=True,
            )

        if skill_names:
            return AgentRouteDecision(
                mode="agent_chain",
                reason="命中 skill，需要按工具链加载规则",
                use_agent_chain=True,
            )

        text = user_input.strip().lower()
        compact = "".join(text.split())
        complex_keywords = (
            "帮我",
            "请",
            "分析",
            "对比",
            "比较",
            "总结",
            "综述",
            "调研",
            "检索",
            "生成",
            "写",
            "设计",
            "方案",
            "报告",
            "规划",
            "拆解",
            "评估",
            "测试",
            "优化",
            "改进",
            "实现",
            "画",
            "整理",
            "推荐",
            "analyze",
            "compare",
            "summarize",
            "survey",
            "search",
            "generate",
            "design",
            "report",
            "plan",
            "evaluate",
            "implement",
        )
        mcp_keywords = ("mcp", "工具", "调用", "server", "服务器")
        has_complex_keyword = any(keyword in compact for keyword in complex_keywords)
        asks_for_mcp = any(keyword in compact for keyword in mcp_keywords)

        mcp_tools = getattr(getattr(self, "mcp_tools", None), "tools", {})
        if asks_for_mcp and mcp_tools:
            return AgentRouteDecision(
                mode="agent_chain",
                reason="用户请求工具调用，且已注册 MCP 工具",
                use_agent_chain=True,
            )

        if use_rag and (has_complex_keyword or len(compact) >= 36):
            return AgentRouteDecision(
                mode="agent_chain",
                reason="科研问题包含多步分析/检索/报告意图",
                use_agent_chain=True,
            )

        if not use_rag and has_complex_keyword and len(compact) >= 18:
            return AgentRouteDecision(
                mode="agent_chain",
                reason="通用问题包含多步执行意图",
                use_agent_chain=True,
            )

        if use_rag:
            return AgentRouteDecision(
                mode="direct_rag",
                reason="简单科研问题，直接 RAG 证据检索后回答",
                use_agent_chain=False,
            )

        return AgentRouteDecision(
            mode="direct_answer",
            reason="普通问题，直接回答",
            use_agent_chain=False,
        )

    def _should_use_lightweight_reply(self, user_input: str) -> bool:
        if self.runtime_documents:
            return False

        text = user_input.strip().lower()
        compact = "".join(text.split())
        if not compact or len(compact) > 24:
            return False

        task_keywords = (
            "论文",
            "文献",
            "研究",
            "方法",
            "模型",
            "实验",
            "分析",
            "总结",
            "对比",
            "检索",
            "上传",
            "文档",
            "rag",
            "paper",
            "survey",
            "experiment",
            "method",
            "analyze",
            "compare",
            "summarize",
        )
        if any(keyword in compact for keyword in task_keywords):
            return False

        lightweight_inputs = {
            "你好",
            "您好",
            "hello",
            "hi",
            "hey",
            "在吗",
            "在不在",
            "谢谢",
            "多谢",
            "感谢",
            "ok",
            "okay",
            "好的",
            "好",
            "嗯",
            "是的",
            "不是",
            "继续",
            "可以",
        }

        return compact in lightweight_inputs

    @staticmethod
    def _build_lightweight_reply(user_input: str) -> str:
        text = "".join(user_input.strip().lower().split())

        if text in {"谢谢", "多谢", "感谢"}:
            return "不客气。"

        if text in {"ok", "okay", "好的", "好", "嗯", "是的", "可以"}:
            return "好的。"

        if text == "继续":
            return "可以，继续说。"

        return "你好，我在。"

    def extract_long_term_memories(
        self,
        user_input: str,
        assistant_answer: str,
    ) -> List[str]:
        """
        从本轮对话中提取值得长期保存的用户偏好、背景和稳定事实。
        """
        prompt = (
            "请从下面这轮对话中提取值得长期记忆的信息。只保留用户稳定偏好、"
            "长期研究方向、项目背景、明确要求或反复有用的事实；不要保存一次性问题、"
            "临时寒暄、模型自己的推测。\n\n"
            "请严格返回 JSON 数组，数组元素是简短中文字符串；如果没有值得保存的信息，返回 []。\n\n"
            f"用户：{user_input}\n\n"
            f"助手：{assistant_answer[:3000]}"
        )

        messages = [
            {
                "role": "system",
                "content": "你是记忆提取器，只输出 JSON 数组，不输出解释。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.client.chat_with_messages(
            messages=messages,
            stream=False,
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        memories = []
        for item in data:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    memories.append(text[:500])

        return memories[:5]

    def _available_agent_tools(self) -> List[AgentToolSpec]:
        tools = [
            AgentToolSpec(
                name="memory_search",
                description="检索跨会话长期记忆，补充用户偏好、历史任务和项目背景。",
            ),
            AgentToolSpec(
                name="document_context",
                description="读取用户本轮上传并已解析的文档文本，作为临时上下文输入 LLM。",
            ),
            AgentToolSpec(
                name="skill_lookup",
                description="加载与当前任务匹配的 skill 规则，作为本轮系统上下文。",
            ),
            AgentToolSpec(
                name="rag_search",
                description="检索本地论文库，执行混合召回、rerank、证据覆盖校验和拒答门控。",
            ),
            AgentToolSpec(
                name="context_summary",
                description="整理当前对话和任务上下文，供最终生成使用。",
            ),
            AgentToolSpec(
                name="final_report",
                description="根据已执行步骤、证据和上下文生成最终回答或报告。",
            ),
        ]
        tools.extend(self.mcp_tools.list_tool_specs())
        return tools

    @staticmethod
    def _build_agent_trace_messages(
        results: List[AgentStepResult],
    ) -> List[Dict[str, str]]:
        trace_text = DynamicAgentPlanner.format_trace(results)
        if not trace_text:
            return []

        return [
            {
                "role": "system",
                "content": (
                    "以下是本轮 Agent 动态计划执行轨迹。最终回答需要结合这些工具结果；"
                    "如果 RAG 证据门控未通过，不得绕过证据约束自由生成。\n"
                    f"{trace_text}"
                ),
            }
        ]

    def _build_runtime_document_messages(self) -> List[Dict[str, str]]:
        if not self.runtime_documents:
            return []

        chunks = []
        total_chars = 0
        max_total_chars = 14_000

        for index, document in enumerate(self.runtime_documents, 1):
            text = document.get("text", "")
            if not text:
                continue

            header = (
                f"【上传文档{index}】\n"
                f"Document ID: {document.get('id', 'N/A')}\n"
                f"Filename: {document.get('filename', 'unknown')}\n"
                "Parsed Text:\n"
            )
            budget = max_total_chars - total_chars
            if budget <= 0:
                break

            available = max(0, budget - len(header) - 20)
            snippet = text[:available]
            if len(text) > len(snippet):
                snippet += "\n...[文档内容已截断]"

            chunk = header + snippet
            chunks.append(chunk)
            total_chars += len(chunk)

        if not chunks:
            return []

        return [
            {
                "role": "system",
                "content": (
                    "以下是用户本轮上传并解析出的文档内容。"
                    "回答时可以基于这些文档内容，但不要把文档内容写入长期记忆；"
                    "如果文档中没有提到，请说明“上传文档中未提到”。\n\n"
                    + "\n\n".join(chunks)
                ),
            }
        ]

    def _build_plan(
        self,
        user_input: str,
        skill_names: List[str],
        use_rag: bool,
    ) -> AgentPlan:
        route_decision = getattr(self, "_last_route_decision", None)
        if route_decision is not None and not route_decision.use_agent_chain:
            return AgentPlan(
                user_input=user_input,
                skill_names=skill_names,
                use_rag=use_rag,
                steps=[route_decision.mode],
            )

        dynamic_plan = self.dynamic_planner.generate_plan(
            user_input=user_input,
            skill_names=skill_names,
            use_rag=use_rag,
            tools=self._available_agent_tools(),
        )
        if self.runtime_documents:
            dynamic_plan.steps = self.dynamic_planner._ensure_tool_before_final(
                steps=dynamic_plan.steps,
                tool="document_context",
                step=DynamicPlanStep(
                    id="doc_context",
                    goal="读取并注入用户上传文档解析内容",
                    tool="document_context",
                    query=",".join(
                        str(document.get("filename", "unknown"))
                        for document in self.runtime_documents
                    ),
                    expected_output="上传文档解析文本",
                ),
            )

        return AgentPlan(
            user_input=user_input,
            skill_names=skill_names,
            use_rag=use_rag,
            steps=[step.goal for step in dynamic_plan.steps],
            dynamic_plan=dynamic_plan,
        )

        steps = [
            "识别用户意图和任务类型",
        ]

        if skill_names:
            steps.append("加载并应用匹配的 skill 规则")
        else:
            steps.append("未命中特定 skill，使用通用科研助手规则")

        if use_rag:
            steps.extend([
                "执行 RAG 检索：BM25 关键词召回 + BGE-M3/向量召回 + RRF 融合",
                "使用 reranker 和相关性门控筛选证据片段",
                "拆解问题需求点并检查证据覆盖",
                "若证据不足则补充检索；仍不足则拒答",
                "生成前再次校验证据覆盖，覆盖不足则停止生成",
                "仅基于证据约束生成带来源回答",
            ])
        else:
            steps.extend([
                "判断该问题不需要本地论文库 RAG",
                "基于系统提示、skill 和对话上下文生成回答",
                "信息不足时如实说明缺失项",
            ])

        return AgentPlan(
            user_input=user_input,
            skill_names=skill_names,
            use_rag=use_rag,
            steps=steps,
        )

    @staticmethod
    def _format_plan(plan: AgentPlan) -> str:
        if plan.dynamic_plan is not None:
            return DynamicAgentPlanner.format_plan(plan.dynamic_plan)

        lines = ["▸ 执行计划："]

        for index, step in enumerate(plan.steps, 1):
            lines.append(f"{index}. {step}")

        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _format_rag_verification(rag_result: RagRetrievalResult) -> str:
        assessment = rag_result.assessment
        lines = [
            "▸ 证据校验结果：",
            f"- 状态：{rag_result.status}",
            f"- 判断：{assessment.reason}",
        ]

        if assessment.requirements:
            lines.append("- 需求覆盖：")
            for requirement in assessment.requirements:
                covered = assessment.coverage.get(requirement, False)
                lines.append(f"  - {requirement}: {'已覆盖' if covered else '未覆盖'}")

        if assessment.missing_requirements:
            lines.append("- 缺失证据点：" + "、".join(assessment.missing_requirements))

        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _build_refusal_answer(rag_result: RagRetrievalResult) -> str:
        missing_text = ""
        if rag_result.assessment.missing_requirements:
            missing_text = (
                "\n\n资料中未提到："
                + "、".join(rag_result.assessment.missing_requirements)
            )

        return (
            "当前本地论文库证据不足，我不能基于这些材料可靠回答这个问题。"
            f"\n\n原因：{rag_result.assessment.reason}"
            f"{missing_text}"
            "\n\n你可以补充更具体的论文标题、关键词，或上传相关文档后再问。"
        )

    @staticmethod
    def _build_hook_refusal(hook_result: HookRunResult) -> str:
        reasons = [
            decision.reason
            for decision in hook_result.blocks
        ]

        if not reasons:
            reasons = [
                decision.reason
                for decision in hook_result.warnings
            ]

        reason_text = "；".join(reasons) if reasons else "安全策略未通过"

        return (
            "当前请求或生成结果未通过安全 Hook 检查，已停止执行。"
            f"\n\n原因：{reason_text}"
        )

    def _pre_generation_evidence_check(
        self,
        rag_result: Optional[RagRetrievalResult],
    ) -> Optional[str]:
        if rag_result is None:
            return None

        assessment = rag_result.assessment
        missing_requirements = [
            requirement
            for requirement, covered in assessment.coverage.items()
            if not covered
        ]

        if rag_result.status != "PASS" or missing_requirements:
            if missing_requirements and not assessment.missing_requirements:
                assessment.missing_requirements = missing_requirements

            logger.info(
                "生成前证据覆盖校验未通过，status=%s, missing=%s",
                rag_result.status,
                assessment.missing_requirements,
            )

            return self._build_refusal_answer(rag_result)

        logger.info("生成前证据覆盖校验通过")
        return None

    def get_summary(self) -> str:
        """
        获取当前历史摘要。
        """
        return self.memory.summary

    def get_available_skills(self) -> List[str]:
        """
        获取当前可用 skill。
        """
        return self.skill_loader.available_skills()

    def clear_skill_cache(self) -> None:
        """
        清空 skill 文件缓存。
        """
        self.skill_loader.clear_cache()
    def reload_rag_from_records(self, records) -> None:
        """
        Web 后端使用：从 MySQL 查询结果重新加载 RAG。

        records 通常来自：
        apps.papers.models.PaperRecord.objects.all()
        """
        self.paper_rag.reload_from_records(records)
    def reload_rag(self) -> None:
        """
        重新加载本地论文库。

        修改 PaperLibrary/papers.json 后，可以调用该方法刷新。
        """
        self.paper_rag.reload()

    def get_paper_count(self) -> int:
        """
        返回本地论文库论文数量。
        """
        return self.paper_rag.count()

    def _should_use_paper_rag(
        self,
        user_input: str,
        skill_names: List[str],
    ) -> bool:
        """
        判断本轮是否需要启用本地论文库 RAG。

        RAG 的定位：
        1. 不是简单关键词匹配。
        2. 不是只有用户明确说“论文”才使用。
        3. 只要问题涉及科研方向、模型方法、技术路线、实验设计、创新点等，
           就可以尝试使用本地论文库增强回答。

        注意：
        这里仅判断是否“尝试 RAG”。
        真正是否注入论文摘要，由 PaperRAGRetriever 是否检索到相关论文决定。
        """

        # 这些 skill 通常需要论文背景增强
        rag_related_skills = {
            "paper_search",
            "paper_reading",
            "research_innovation",
        }

        if any(skill in rag_related_skills for skill in skill_names):
            return True

        text = user_input.lower()

        research_keywords = (
            # 中文科研相关
            "论文",
            "文献",
            "研究",
            "方向",
            "领域",
            "方法",
            "模型",
            "算法",
            "实验",
            "创新",
            "创新点",
            "综述",
            "数据集",
            "基准",
            "对比实验",
            "消融实验",
            "技术路线",

            # 英文科研相关
            "paper",
            "survey",
            "method",
            "model",
            "architecture",
            "experiment",
            "benchmark",
            "dataset",
            "sota",

            # 常见模型 / 技术关键词
            "rag",
            "mamba",
            "transformer",
            "diffusion",
            "attention",
            "cnn",
            "gnn",
            "llm",
            "bert",
            "clip",
        )

        if any(keyword in text for keyword in research_keywords):
            return True

        return False

    def _record_auto_memory(
        self,
        user_input: str,
        assistant_answer: str,
    ) -> None:
        try:
            self.auto_memory.record_turn(
                user_input=user_input,
                assistant_answer=assistant_answer,
                metadata={"source": "ResearchAgent.ask_stream"},
                summary_client=self.client,
            )
        except Exception:
            logger.exception("Auto Memory 写入失败，已忽略")

    def _build_long_term_memory_messages(
        self,
        user_input: str = "",
    ) -> List[Dict[str, str]]:
        file_memories = self.auto_memory.search(user_input, limit=20)
        all_memories = []

        for memory in self.long_term_memories + file_memories:
            memory = memory.strip()
            if memory and memory not in all_memories:
                all_memories.append(memory)

        if not all_memories:
            return []

        memory_text = "\n".join(
            f"- {memory}"
            for memory in all_memories[:20]
        )

        return [
            {
                "role": "system",
                "content": (
                    "以下是关于用户的长期记忆，包括稳定偏好、研究背景和项目上下文。"
                    "回答时请自然参考，不要生硬复述：\n"
                    f"{memory_text}"
                ),
            }
        ]

    @staticmethod
    def _inject_extra_system_messages(
        messages: List[Dict[str, str]],
        extra_messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        将本轮额外 system messages 临时插入 messages。

        extra_messages 包括：
        1. skill.md 内容
        2. RAG 论文摘要

        插入位置：
        system prompt
        + extra messages
        + summary
        + recent messages
        + current user

        注意：
        extra_messages 不写入 memory，只在当前轮生效。
        """

        if not extra_messages:
            return messages

        if messages and messages[0]["role"] == "system":
            return [messages[0]] + extra_messages + messages[1:]

        return extra_messages + messages
