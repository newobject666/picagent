from types import SimpleNamespace

from figure_agent.agent.research_agent import ResearchAgent
from figure_agent.agent.runner import AgentExecutionTrace, AgentExecutor, AgentPlan, AgentRunner


def _agent_without_init() -> ResearchAgent:
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.runtime_documents = []
    agent.mcp_tools = SimpleNamespace(tools={})
    return agent


def test_simple_general_question_uses_direct_answer_route():
    agent = _agent_without_init()

    decision = agent._decide_agent_route(
        user_input="What is your name?",
        skill_names=[],
        use_rag=False,
    )

    assert decision.mode == "direct_answer"
    assert not decision.use_agent_chain


def test_simple_research_question_uses_direct_rag_route():
    agent = _agent_without_init()

    decision = agent._decide_agent_route(
        user_input="What is CNN?",
        skill_names=[],
        use_rag=True,
    )

    assert decision.mode == "direct_rag"
    assert not decision.use_agent_chain


def test_complex_research_task_triggers_agent_chain():
    agent = _agent_without_init()

    decision = agent._decide_agent_route(
        user_input="Please compare Transformer and Mamba papers and generate a report",
        skill_names=[],
        use_rag=True,
    )

    assert decision.mode == "agent_chain"
    assert decision.use_agent_chain


def test_uploaded_document_triggers_agent_chain():
    agent = _agent_without_init()
    agent.runtime_documents = [{"filename": "note.md", "text": "content"}]

    decision = agent._decide_agent_route(
        user_input="hello",
        skill_names=[],
        use_rag=False,
    )

    assert decision.mode == "agent_chain"
    assert decision.use_agent_chain


def test_direct_route_build_plan_does_not_call_dynamic_planner():
    class FailingPlanner:
        def generate_plan(self, *args, **kwargs):
            raise AssertionError("dynamic planner should not run for direct routes")

    agent = _agent_without_init()
    agent.dynamic_planner = FailingPlanner()
    agent._last_route_decision = SimpleNamespace(
        mode="direct_answer",
        use_agent_chain=False,
    )

    plan = agent._build_plan(
        user_input="What is your name?",
        skill_names=[],
        use_rag=False,
    )

    assert plan.steps == ["direct_answer"]
    assert plan.dynamic_plan is None


def test_agent_trace_is_hidden_by_default():
    assert ResearchAgent._should_show_agent_trace() is False


def test_runner_module_exposes_executor_and_runner_alias():
    assert issubclass(AgentRunner, AgentExecutor)


def test_agent_executor_handles_direct_answer_without_tools():
    agent = _agent_without_init()
    agent._last_route_decision = SimpleNamespace(use_agent_chain=False)
    executor = AgentExecutor(agent)
    plan = AgentPlan(
        user_input="What is your name?",
        skill_names=[],
        use_rag=False,
        steps=["direct_answer"],
    )

    trace = executor.execute(
        user_input="What is your name?",
        plan=plan,
        skill_names=[],
    )

    assert isinstance(trace, AgentExecutionTrace)
    assert trace.results == []
