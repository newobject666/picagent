import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from figure_agent.rag.accuracy_tester import RAGAccuracyTester


CASE_PATH = Path(__file__).with_name("rag_accuracy_cases.json")


RAG_CONTEXT = """
本轮问题触发了本地论文库 RAG 防幻觉策略。

【证据1】
Title：Kafka 消息堆积处理
Summary：Kafka 消息堆积可以通过扩容消费者和优化消费速率处理。

【证据2】
Title：Kafka 消息可靠性和不丢机制
Summary：Kafka 通过 acks、副本、ISR、生产者重试和消费位移提交来增强消息可靠性，降低消息不丢风险。
"""


def test_supported_answer_passes():
    tester = RAGAccuracyTester()
    answer = (
        "Kafka 消息堆积可以通过扩容消费者和优化消费速率处理。[来源: 证据1]\n"
        "Kafka 可以通过 acks、副本、ISR、生产者重试和消费位移提交增强消息可靠性。[来源: 证据2]"
    )

    report = tester.evaluate(answer=answer, rag_context=RAG_CONTEXT)

    assert report.status == "PASS"
    assert report.total_claims == 2
    assert report.passed_claims == 2


def test_claim_without_source_fails():
    tester = RAGAccuracyTester()
    answer = "Kafka 消息堆积可以通过扩容消费者和优化消费速率处理。"

    report = tester.evaluate(answer=answer, rag_context=RAG_CONTEXT)

    assert report.status == "FAIL"
    assert report.missing_source_claims


def test_unsupported_claim_fails():
    tester = RAGAccuracyTester()
    answer = "Kafka 可以通过量子加密协议保证消息永远不丢。[来源: 证据2]"

    report = tester.evaluate(answer=answer, rag_context=RAG_CONTEXT)

    assert report.status == "FAIL"
    assert report.unsupported_claims


def test_uncertainty_answer_passes_without_source():
    tester = RAGAccuracyTester()
    answer = "资料中未提到 Kafka 是否使用量子加密协议保证消息不丢。"

    report = tester.evaluate(answer=answer, rag_context=RAG_CONTEXT)

    assert report.status == "PASS"


def load_accuracy_cases():
    with open(CASE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_context_from_case(case):
    chunks = []

    for index, evidence in enumerate(case["gold_evidence"], start=1):
        chunks.append(
            f"【证据{index}】\n"
            f"Question：{case['question']}\n"
            f"Summary：{evidence}"
        )

    return "\n\n".join(chunks)


def build_answer_from_case(case):
    if case.get("should_refuse"):
        return "资料中未提到足够信息，无法基于当前证据可靠回答。"

    lines = []

    for claim in case["gold_claims"]:
        source_index = 1

        for index, evidence in enumerate(case["gold_evidence"], start=1):
            if claim in evidence:
                source_index = index
                break

        lines.append(f"{claim}。[来源: 证据{source_index}]")

    return "\n".join(lines)


def test_rag_accuracy_cases_dataset():
    tester = RAGAccuracyTester()
    cases = load_accuracy_cases()

    assert len(cases) == 20

    for case in cases:
        answer = build_answer_from_case(case)
        context = build_context_from_case(case)
        report = tester.evaluate(answer=answer, rag_context=context)

        expected_status = "PASS"
        assert report.status == expected_status, (
            f"case failed: {case['question']}\n"
            f"status={report.status}\n"
            f"unsupported={report.unsupported_claims}\n"
            f"missing_source={report.missing_source_claims}"
        )


if __name__ == "__main__":
    test_supported_answer_passes()
    test_claim_without_source_fails()
    test_unsupported_claim_fails()
    test_uncertainty_answer_passes_without_source()
    test_rag_accuracy_cases_dataset()
    print("RAG accuracy tests passed.")
