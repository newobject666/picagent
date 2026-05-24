# figure_agent/agent/runner.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from figure_agent.agent.dynamic_planner import (
    AgentStepResult,
    DynamicPlan,
    DynamicPlanStep,
)
from figure_agent.agent.hooks import HookContext, HookEvent
from figure_agent.rag.paper_retriever import RagRetrievalResult


@dataclass
class AgentPlan:
    user_input: str
    skill_names: List[str]
    use_rag: bool
    steps: List[str]
    dynamic_plan: Optional[DynamicPlan] = None


@dataclass
class AgentExecutionTrace:
    results: List[AgentStepResult]
    rag_result: Optional[RagRetrievalResult] = None
    rag_messages: List[Dict[str, str]] = None
    skill_messages: List[Dict[str, str]] = None
    user_updates: List[str] = None

    def __post_init__(self) -> None:
        self.rag_messages = self.rag_messages or []
        self.skill_messages = self.skill_messages or []
        self.user_updates = self.user_updates or []


class AgentExecutor:
    """Execute direct and dynamic Agent plans.

    The LLM planner decides the steps. This executor owns the deterministic
    runtime loop: validate the route, call tools, collect observations, and ask
    the planner to revise remaining steps after each result.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def execute(
        self,
        user_input: str,
        plan: AgentPlan,
        skill_names: List[str],
    ) -> AgentExecutionTrace:
        route_decision = getattr(self.agent, "_last_route_decision", None)
        if route_decision is not None and not route_decision.use_agent_chain:
            return self.execute_direct_path(
                user_input=user_input,
                plan=plan,
                skill_names=skill_names,
                use_rag=plan.use_rag,
            )

        dynamic_plan = plan.dynamic_plan
        if dynamic_plan is None:
            dynamic_plan = self.agent.dynamic_planner.generate_plan(
                user_input=user_input,
                skill_names=skill_names,
                use_rag=plan.use_rag,
                tools=self.agent._available_agent_tools(),
            )
            plan.dynamic_plan = dynamic_plan

        trace = AgentExecutionTrace(results=[])
        tools = self.agent._available_agent_tools()
        step_index = 0
        executed_count = 0

        while step_index < len(dynamic_plan.steps) and executed_count < self.agent.dynamic_planner.max_steps:
            step = dynamic_plan.steps[step_index]
            result = self.execute_step(
                step=step,
                user_input=user_input,
                plan=plan,
                skill_names=skill_names,
                trace=trace,
            )
            trace.results.append(result)
            trace.user_updates.append(self.format_agent_step_update(result))
            executed_count += 1

            if step.tool == "final_report":
                break

            completed = dynamic_plan.steps[: step_index + 1]
            remaining = dynamic_plan.steps[step_index + 1:]
            revised_remaining = self.agent.dynamic_planner.revise_plan(
                user_input=user_input,
                plan=dynamic_plan,
                completed_results=trace.results,
                remaining_steps=remaining,
                tools=tools,
            )
            dynamic_plan.steps = completed + revised_remaining
            plan.steps = [item.goal for item in dynamic_plan.steps]
            step_index += 1

        return trace

    def execute_direct_path(
        self,
        user_input: str,
        plan: AgentPlan,
        skill_names: List[str],
        use_rag: bool,
    ) -> AgentExecutionTrace:
        trace = AgentExecutionTrace(results=[])

        if skill_names:
            trace.skill_messages = self.agent.skill_loader.build_skill_messages(user_input)
            trace.results.append(
                AgentStepResult(
                    step_id="direct_skill",
                    goal="Load matched skills",
                    tool="skill_lookup",
                    query=",".join(skill_names),
                    status="PASS",
                    observation=f"Loaded skills: {', '.join(skill_names)}",
                )
            )

        if use_rag:
            before_rag_result = self.agent.hooks.run(
                HookContext(
                    event=HookEvent.BEFORE_RAG,
                    user_input=user_input,
                    plan=plan,
                    skill_names=skill_names,
                )
            )
            if not before_rag_result.allowed:
                trace.results.append(
                    AgentStepResult(
                        step_id="direct_rag",
                        goal="Direct RAG retrieval",
                        tool="rag_search",
                        query=user_input,
                        status="BLOCKED",
                        observation=self.agent._build_hook_refusal(before_rag_result),
                    )
                )
                return trace

            rag_result = self.agent.paper_rag.retrieve_with_guardrails(query=user_input)
            trace.rag_result = rag_result
            after_rag_result = self.agent.hooks.run(
                HookContext(
                    event=HookEvent.AFTER_RAG,
                    user_input=user_input,
                    plan=plan,
                    skill_names=skill_names,
                    rag_result=rag_result,
                )
            )
            if not after_rag_result.allowed:
                trace.results.append(
                    AgentStepResult(
                        step_id="direct_rag",
                        goal="Direct RAG retrieval",
                        tool="rag_search",
                        query=user_input,
                        status="BLOCKED",
                        observation=self.agent._build_hook_refusal(after_rag_result),
                        evidence_status=rag_result.status,
                    )
                )
                return trace

            if rag_result.context_message:
                trace.rag_messages = [rag_result.context_message]

            trace.results.append(
                AgentStepResult(
                    step_id="direct_rag",
                    goal="Direct RAG retrieval with evidence gate",
                    tool="rag_search",
                    query=user_input,
                    status=rag_result.status,
                    observation=f"RAG evidence gate={rag_result.status}",
                    evidence_status=rag_result.status,
                )
            )

        return trace

    def execute_step(
        self,
        step: DynamicPlanStep,
        user_input: str,
        plan: AgentPlan,
        skill_names: List[str],
        trace: AgentExecutionTrace,
    ) -> AgentStepResult:
        query = step.query or user_input

        if step.tool == "memory_search":
            memories = self.agent.auto_memory.search(query, limit=5)
            observation = (
                f"Found {len(memories)} long-term memory item(s)"
                if memories
                else "No relevant long-term memory found"
            )
            return AgentStepResult(
                step_id=step.id,
                goal=step.goal,
                tool=step.tool,
                query=query,
                status="PASS",
                observation=observation,
            )

        if step.tool == "skill_lookup":
            if not trace.skill_messages:
                trace.skill_messages = self.agent.skill_loader.build_skill_messages(user_input)
            observation = (
                f"Loaded skills: {', '.join(skill_names)}"
                if skill_names
                else "No specific skill matched"
            )
            return AgentStepResult(
                step_id=step.id,
                goal=step.goal,
                tool=step.tool,
                query=query,
                status="PASS",
                observation=observation,
            )

        if step.tool == "document_context":
            document_count = len(self.agent.runtime_documents)
            observation = (
                f"Loaded {document_count} uploaded document(s) for final context"
                if document_count
                else "No uploaded document in this turn"
            )
            return AgentStepResult(
                step_id=step.id,
                goal=step.goal,
                tool=step.tool,
                query=query,
                status="PASS",
                observation=observation,
            )

        if step.tool == "rag_search":
            before_rag_result = self.agent.hooks.run(
                HookContext(
                    event=HookEvent.BEFORE_RAG,
                    user_input=user_input,
                    plan=plan,
                    skill_names=skill_names,
                )
            )
            if not before_rag_result.allowed:
                return AgentStepResult(
                    step_id=step.id,
                    goal=step.goal,
                    tool=step.tool,
                    query=query,
                    status="BLOCKED",
                    observation=self.agent._build_hook_refusal(before_rag_result),
                )

            rag_result = self.agent.paper_rag.retrieve_with_guardrails(query=query)
            trace.rag_result = rag_result
            after_rag_result = self.agent.hooks.run(
                HookContext(
                    event=HookEvent.AFTER_RAG,
                    user_input=user_input,
                    plan=plan,
                    skill_names=skill_names,
                    rag_result=rag_result,
                )
            )
            if not after_rag_result.allowed:
                return AgentStepResult(
                    step_id=step.id,
                    goal=step.goal,
                    tool=step.tool,
                    query=query,
                    status="BLOCKED",
                    observation=self.agent._build_hook_refusal(after_rag_result),
                    evidence_status=rag_result.status,
                )

            if rag_result.context_message:
                trace.rag_messages = [rag_result.context_message]

            if rag_result.supplemental_queries:
                observation = (
                    "Initial evidence was insufficient; supplemental retrieval "
                    f"completed with final status {rag_result.status}"
                )
            else:
                observation = (
                    "Completed hybrid retrieval, rerank, and evidence gate; "
                    f"final status {rag_result.status}"
                )

            return AgentStepResult(
                step_id=step.id,
                goal=step.goal,
                tool=step.tool,
                query=query,
                status=rag_result.status,
                observation=observation,
                evidence_status=rag_result.status,
            )

        if step.tool == "context_summary":
            return AgentStepResult(
                step_id=step.id,
                goal=step.goal,
                tool=step.tool,
                query=query,
                status="PASS",
                observation="Prepared current conversation, memory, skill, and RAG context",
            )

        if step.tool == "final_report":
            return AgentStepResult(
                step_id=step.id,
                goal=step.goal,
                tool=step.tool,
                query=query,
                status="READY",
                observation="Ready for final generation with pre-generation evidence check",
            )

        if step.tool.startswith("mcp:"):
            result_text = self.agent.mcp_tools.call_tool(
                tool_name=step.tool,
                query=query,
                context={
                    "user_input": user_input,
                    "skill_names": skill_names,
                    "documents": [
                        {
                            "filename": document.get("filename"),
                            "char_count": len(document.get("text", "")),
                        }
                        for document in self.agent.runtime_documents
                    ],
                },
            )
            return AgentStepResult(
                step_id=step.id,
                goal=step.goal,
                tool=step.tool,
                query=query,
                status="PASS",
                observation=result_text[:1000],
            )

        return AgentStepResult(
            step_id=step.id,
            goal=step.goal,
            tool=step.tool,
            query=query,
            status="SKIPPED",
            observation=f"Unknown tool {step.tool}; skipped",
        )

    @staticmethod
    def format_agent_step_update(result: AgentStepResult) -> str:
        evidence = f", evidence_status={result.evidence_status}" if result.evidence_status else ""
        return (
            f"Step {result.step_id}: {result.goal}\n"
            f"- tool: {result.tool}\n"
            f"- status: {result.status}{evidence}\n"
            f"- observation: {result.observation}\n\n"
        )


class AgentRunner(AgentExecutor):
    """Compatibility name for the Agent execution loop."""

