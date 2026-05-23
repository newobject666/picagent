import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class AgentToolSpec:
    name: str
    description: str


@dataclass
class DynamicPlanStep:
    id: str
    goal: str
    tool: str
    query: str = ""
    expected_output: str = ""
    status: str = "pending"


@dataclass
class DynamicPlan:
    task: str
    rationale: str
    steps: List[DynamicPlanStep] = field(default_factory=list)


@dataclass
class AgentStepResult:
    step_id: str
    goal: str
    tool: str
    query: str
    status: str
    observation: str
    evidence_status: str = ""


class DynamicAgentPlanner:
    """
    LLM-driven planner for the Agent execution loop.

    The planner is intentionally conservative: it lets the LLM decide the
    sequence, but normalizes every tool choice against a small allow-list and
    falls back to a deterministic plan if the model output is not valid JSON.
    """

    max_steps = 6
    final_tool = "final_report"

    def __init__(self, client):
        self.client = client

    def generate_plan(
        self,
        user_input: str,
        skill_names: Sequence[str],
        use_rag: bool,
        tools: Sequence[AgentToolSpec],
    ) -> DynamicPlan:
        tool_names = {tool.name for tool in tools}
        prompt = self._build_plan_prompt(
            user_input=user_input,
            skill_names=skill_names,
            use_rag=use_rag,
            tools=tools,
        )

        try:
            content = self._call_llm_json(prompt, max_tokens=1200)
            plan = self._parse_plan(content, tool_names)
        except Exception:
            plan = self._fallback_plan(user_input, skill_names, use_rag)

        return self._normalize_plan(
            plan=plan,
            user_input=user_input,
            skill_names=skill_names,
            use_rag=use_rag,
            tool_names=tool_names,
        )

    def revise_plan(
        self,
        user_input: str,
        plan: DynamicPlan,
        completed_results: Sequence[AgentStepResult],
        remaining_steps: Sequence[DynamicPlanStep],
        tools: Sequence[AgentToolSpec],
    ) -> List[DynamicPlanStep]:
        if not remaining_steps:
            return []

        tool_names = {tool.name for tool in tools}
        prompt = self._build_revision_prompt(
            user_input=user_input,
            plan=plan,
            completed_results=completed_results,
            remaining_steps=remaining_steps,
            tools=tools,
        )

        try:
            content = self._call_llm_json(prompt, max_tokens=1000)
            data = self._extract_json(content)
            raw_steps = data.get("remaining_steps") or data.get("steps") or []
            revised = self._parse_steps(raw_steps, tool_names)
        except Exception:
            return list(remaining_steps)

        if not revised:
            return list(remaining_steps)

        return self._ensure_final_step(revised, user_input)[: self.max_steps]

    @classmethod
    def format_plan(cls, plan: DynamicPlan) -> str:
        lines = ["▶ 动态执行计划："]
        if plan.rationale:
            lines.append(f"规划依据：{plan.rationale}")

        for index, step in enumerate(plan.steps, 1):
            query = f"；query={step.query}" if step.query else ""
            lines.append(f"{index}. [{step.tool}] {step.goal}{query}")

        return "\n".join(lines) + "\n\n"

    @staticmethod
    def format_trace(results: Sequence[AgentStepResult]) -> str:
        if not results:
            return ""

        lines = ["Agent 执行轨迹："]
        for index, result in enumerate(results, 1):
            evidence = f"; evidence={result.evidence_status}" if result.evidence_status else ""
            lines.append(
                f"{index}. [{result.status}] tool={result.tool}; goal={result.goal}{evidence}\n"
                f"   observation={result.observation}"
            )

        return "\n".join(lines)

    def _call_llm_json(self, prompt: str, max_tokens: int) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 Agent 规划器。只输出 JSON，不输出 Markdown，不解释。"
                    "所有 tool 必须来自用户给定的工具列表。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        response = self.client.chat_with_messages(
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _build_plan_prompt(
        self,
        user_input: str,
        skill_names: Sequence[str],
        use_rag: bool,
        tools: Sequence[AgentToolSpec],
    ) -> str:
        tool_text = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in tools
        )
        return (
            "请为科研 Agent 生成一个动态 Plan-to-Solve。\n"
            "要求：先拆解任务，再为每一步选择一个工具；最后一步必须是 final_report。\n"
            "如果需要论文证据，必须包含 rag_search；如果需要历史上下文，可使用 memory_search；"
            "如果命中技能，可使用 skill_lookup。\n\n"
            "返回 JSON 格式：\n"
            "{\n"
            '  "task": "用户任务",\n'
            '  "rationale": "为什么这样规划",\n'
            '  "steps": [\n'
            '    {"id": "s1", "goal": "步骤目标", "tool": "工具名", "query": "工具输入", "expected_output": "期望结果"}\n'
            "  ]\n"
            "}\n\n"
            f"用户问题：{user_input}\n"
            f"命中技能：{', '.join(skill_names) if skill_names else '无'}\n"
            f"是否需要论文 RAG：{use_rag}\n"
            f"可用工具：\n{tool_text}"
        )

    def _build_revision_prompt(
        self,
        user_input: str,
        plan: DynamicPlan,
        completed_results: Sequence[AgentStepResult],
        remaining_steps: Sequence[DynamicPlanStep],
        tools: Sequence[AgentToolSpec],
    ) -> str:
        tool_text = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in tools
        )
        completed_text = "\n".join(
            f"- {result.step_id}: tool={result.tool}, status={result.status}, "
            f"evidence={result.evidence_status}, observation={result.observation}"
            for result in completed_results
        )
        remaining_text = "\n".join(
            f"- {step.id}: tool={step.tool}, goal={step.goal}, query={step.query}"
            for step in remaining_steps
        )
        return (
            "请根据已执行结果调整剩余计划。只返回 JSON。\n"
            "如果现有剩余步骤仍然合适，就原样返回；如果证据不足，可以增加或改写 rag_search；"
            "最后一步必须保留 final_report。\n\n"
            "返回格式：\n"
            '{"remaining_steps": [{"id": "s2", "goal": "步骤目标", "tool": "工具名", '
            '"query": "工具输入", "expected_output": "期望结果"}]}\n\n'
            f"用户问题：{user_input}\n"
            f"原始计划：{plan.rationale}\n"
            f"已完成步骤：\n{completed_text or '无'}\n\n"
            f"剩余步骤：\n{remaining_text or '无'}\n\n"
            f"可用工具：\n{tool_text}"
        )

    def _parse_plan(self, content: str, tool_names: set[str]) -> DynamicPlan:
        data = self._extract_json(content)
        steps = self._parse_steps(data.get("steps", []), tool_names)
        return DynamicPlan(
            task=str(data.get("task", "")).strip(),
            rationale=str(data.get("rationale", "")).strip(),
            steps=steps,
        )

    def _parse_steps(self, raw_steps, tool_names: set[str]) -> List[DynamicPlanStep]:
        if not isinstance(raw_steps, list):
            return []

        steps = []
        for index, item in enumerate(raw_steps, 1):
            if not isinstance(item, dict):
                continue

            tool = str(item.get("tool", "")).strip()
            if tool not in tool_names:
                continue

            goal = str(item.get("goal", "")).strip()
            if not goal:
                continue

            step_id = str(item.get("id") or f"s{index}").strip()
            steps.append(
                DynamicPlanStep(
                    id=step_id,
                    goal=goal[:300],
                    tool=tool,
                    query=str(item.get("query", "")).strip()[:500],
                    expected_output=str(item.get("expected_output", "")).strip()[:300],
                )
            )

        return steps[: self.max_steps]

    def _normalize_plan(
        self,
        plan: DynamicPlan,
        user_input: str,
        skill_names: Sequence[str],
        use_rag: bool,
        tool_names: set[str],
    ) -> DynamicPlan:
        if not plan.task:
            plan.task = user_input
        if not plan.rationale:
            plan.rationale = "根据用户问题、可用工具和证据约束动态规划执行步骤"
        if not plan.steps:
            plan = self._fallback_plan(user_input, skill_names, use_rag)

        plan.steps = [
            step
            for step in plan.steps
            if step.tool in tool_names
        ][: self.max_steps]

        if use_rag and "rag_search" in tool_names:
            plan.steps = self._ensure_tool_before_final(
                steps=plan.steps,
                tool="rag_search",
                step=DynamicPlanStep(
                    id="rag_required",
                    goal="检索论文库并进行证据门控",
                    tool="rag_search",
                    query=user_input,
                    expected_output="带证据状态的论文检索结果",
                ),
            )

        if skill_names and "skill_lookup" in tool_names:
            plan.steps = self._ensure_tool_before_final(
                steps=plan.steps,
                tool="skill_lookup",
                step=DynamicPlanStep(
                    id="skill_required",
                    goal="加载与任务匹配的技能规则",
                    tool="skill_lookup",
                    query=",".join(skill_names),
                    expected_output="本轮需要注入的技能说明",
                ),
            )

        plan.steps = self._ensure_final_step(plan.steps, user_input)
        plan.steps = plan.steps[: self.max_steps]
        return plan

    def _fallback_plan(
        self,
        user_input: str,
        skill_names: Sequence[str],
        use_rag: bool,
    ) -> DynamicPlan:
        steps = [
            DynamicPlanStep(
                id="s1",
                goal="检索长期记忆，补充用户偏好和历史任务上下文",
                tool="memory_search",
                query=user_input,
                expected_output="相关历史记忆片段",
            )
        ]

        if skill_names:
            steps.append(
                DynamicPlanStep(
                    id="s2",
                    goal="加载命中的技能规则",
                    tool="skill_lookup",
                    query=",".join(skill_names),
                    expected_output="技能说明",
                )
            )

        if use_rag:
            steps.append(
                DynamicPlanStep(
                    id=f"s{len(steps) + 1}",
                    goal="检索论文库并校验证据覆盖",
                    tool="rag_search",
                    query=user_input,
                    expected_output="可支撑回答的证据片段或拒答原因",
                )
            )
        else:
            steps.append(
                DynamicPlanStep(
                    id=f"s{len(steps) + 1}",
                    goal="整理当前对话和任务上下文",
                    tool="context_summary",
                    query=user_input,
                    expected_output="生成前上下文摘要",
                )
            )

        steps.append(
            DynamicPlanStep(
                id=f"s{len(steps) + 1}",
                goal="根据执行结果生成最终报告",
                tool=self.final_tool,
                query=user_input,
                expected_output="最终回答或报告",
            )
        )
        return DynamicPlan(
            task=user_input,
            rationale="LLM 计划不可用时使用保守回退计划",
            steps=steps[: self.max_steps],
        )

    @staticmethod
    def _extract_json(content: str) -> Dict:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _ensure_tool_before_final(
        self,
        steps: List[DynamicPlanStep],
        tool: str,
        step: DynamicPlanStep,
    ) -> List[DynamicPlanStep]:
        if any(item.tool == tool for item in steps):
            return steps

        final_steps = [item for item in steps if item.tool == self.final_tool]
        body = [item for item in steps if item.tool != self.final_tool]
        return body + [step] + final_steps

    def _ensure_final_step(
        self,
        steps: List[DynamicPlanStep],
        user_input: str,
    ) -> List[DynamicPlanStep]:
        body = [step for step in steps if step.tool != self.final_tool]
        final_steps = [step for step in steps if step.tool == self.final_tool]

        if final_steps:
            return body + [final_steps[-1]]

        return body + [
            DynamicPlanStep(
                id=f"s{len(body) + 1}",
                goal="根据执行结果生成最终报告",
                tool=self.final_tool,
                query=user_input,
                expected_output="最终回答或报告",
            )
        ]
