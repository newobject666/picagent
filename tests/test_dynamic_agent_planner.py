import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from figure_agent.agent.dynamic_planner import (
    AgentStepResult,
    AgentToolSpec,
    DynamicAgentPlanner,
    DynamicPlan,
    DynamicPlanStep,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakePlannerClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat_with_messages(self, messages, stream=False, max_tokens=None):
        self.calls.append(messages)
        content = self.outputs.pop(0)
        return _FakeResponse(content)


def _tools():
    return [
        AgentToolSpec("memory_search", "search memory"),
        AgentToolSpec("skill_lookup", "load skill"),
        AgentToolSpec("rag_search", "search papers"),
        AgentToolSpec("context_summary", "summarize context"),
        AgentToolSpec("final_report", "final report"),
    ]


def test_llm_dynamic_plan_is_parsed_and_normalized():
    client = FakePlannerClient([
        json.dumps(
            {
                "task": "compare transformer papers",
                "rationale": "need evidence",
                "steps": [
                    {
                        "id": "s1",
                        "goal": "search memory",
                        "tool": "memory_search",
                        "query": "transformer",
                    },
                    {
                        "id": "s2",
                        "goal": "write report",
                        "tool": "final_report",
                    },
                ],
            }
        )
    ])
    planner = DynamicAgentPlanner(client)

    plan = planner.generate_plan(
        user_input="对比 Transformer 论文",
        skill_names=[],
        use_rag=True,
        tools=_tools(),
    )

    assert [step.tool for step in plan.steps] == [
        "memory_search",
        "rag_search",
        "final_report",
    ]
    assert plan.steps[-1].tool == "final_report"


def test_invalid_llm_plan_falls_back_to_safe_plan():
    planner = DynamicAgentPlanner(FakePlannerClient(["not json"]))

    plan = planner.generate_plan(
        user_input="总结 RAG 方法",
        skill_names=["paper_search"],
        use_rag=True,
        tools=_tools(),
    )

    tools = [step.tool for step in plan.steps]
    assert "memory_search" in tools
    assert "skill_lookup" in tools
    assert "rag_search" in tools
    assert tools[-1] == "final_report"


def test_plan_revision_can_replace_remaining_steps_after_observation():
    client = FakePlannerClient([
        json.dumps(
            {
                "remaining_steps": [
                    {
                        "id": "s2b",
                        "goal": "supplement evidence",
                        "tool": "rag_search",
                        "query": "missing reliability evidence",
                    },
                    {
                        "id": "s3",
                        "goal": "final answer",
                        "tool": "final_report",
                    },
                ]
            }
        )
    ])
    planner = DynamicAgentPlanner(client)
    plan = DynamicPlan(
        task="Kafka 可靠性",
        rationale="need evidence",
        steps=[
            DynamicPlanStep("s1", "initial rag", "rag_search", "Kafka"),
            DynamicPlanStep("s2", "final", "final_report", "Kafka"),
        ],
    )
    results = [
        AgentStepResult(
            step_id="s1",
            goal="initial rag",
            tool="rag_search",
            query="Kafka",
            status="RETRY",
            observation="missing reliability evidence",
            evidence_status="RETRY",
        )
    ]

    revised = planner.revise_plan(
        user_input="Kafka 如何保证不丢消息",
        plan=plan,
        completed_results=results,
        remaining_steps=plan.steps[1:],
        tools=_tools(),
    )

    assert [step.tool for step in revised] == ["rag_search", "final_report"]
    assert revised[0].query == "missing reliability evidence"
