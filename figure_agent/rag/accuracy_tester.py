import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class EvidenceChunk:
    evidence_id: str
    text: str


@dataclass
class ClaimCheck:
    claim: str
    source_ids: List[str]
    status: str
    support_score: float
    reason: str


@dataclass
class AccuracyReport:
    status: str
    overall_score: float
    total_claims: int
    passed_claims: int
    checks: List[ClaimCheck] = field(default_factory=list)
    missing_source_claims: List[str] = field(default_factory=list)
    unknown_source_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def raise_for_failure(self) -> None:
        if self.status == "PASS":
            return

        details = [
            f"Accuracy check failed: status={self.status}, score={self.overall_score:.2f}",
        ]

        for check in self.checks:
            if check.status != "PASS":
                details.append(f"- {check.status}: {check.claim} ({check.reason})")

        raise AssertionError("\n".join(details))


class RAGAccuracyTester:
    """
    Verify whether a RAG answer is faithful to provided evidence chunks.

    It checks:
    1. Every factual claim has a source marker, e.g. [来源: 证据1].
    2. Referenced evidence IDs exist.
    3. Referenced evidence text supports the claim.
    """

    def __init__(
        self,
        min_support_score: float = 0.32,
        min_overall_score: float = 1.0,
        embedding_dim: int = 384,
    ):
        self.min_support_score = min_support_score
        self.min_overall_score = min_overall_score
        self.embedding_dim = embedding_dim

    def evaluate(
        self,
        answer: str,
        evidence_chunks: Optional[Sequence[EvidenceChunk]] = None,
        rag_context: Optional[str] = None,
    ) -> AccuracyReport:
        chunks = list(evidence_chunks or [])

        if rag_context:
            chunks.extend(self.parse_evidence_chunks(rag_context))

        chunk_by_id = {
            self._normalize_evidence_id(chunk.evidence_id): chunk
            for chunk in chunks
        }
        claims = self.extract_claims(answer)
        checks: List[ClaimCheck] = []
        missing_source_claims: List[str] = []
        unknown_source_claims: List[str] = []
        unsupported_claims: List[str] = []

        if not chunks:
            return AccuracyReport(
                status="FAIL",
                overall_score=0.0,
                total_claims=len(claims),
                passed_claims=0,
                notes=["没有提供证据片段，无法验证回答准确性。"],
            )

        if not claims:
            status = "PASS" if self._has_uncertainty_answer(answer) else "WARN"
            return AccuracyReport(
                status=status,
                overall_score=1.0 if status == "PASS" else 0.0,
                total_claims=0,
                passed_claims=0,
                notes=["没有抽取到需要验证的事实性结论。"],
            )

        for claim in claims:
            if self._is_uncertainty_statement(claim):
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        source_ids=[],
                        status="PASS",
                        support_score=1.0,
                        reason="不确定性/资料缺失声明，不要求来源。",
                    )
                )
                continue

            source_ids = self.extract_source_ids(claim)

            if not source_ids:
                missing_source_claims.append(claim)
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        source_ids=[],
                        status="FAIL",
                        support_score=0.0,
                        reason="事实性结论缺少 [来源: 证据N] 标注。",
                    )
                )
                continue

            unknown_sources = [
                source_id
                for source_id in source_ids
                if self._normalize_evidence_id(source_id) not in chunk_by_id
            ]

            if unknown_sources:
                unknown_source_claims.append(claim)
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        source_ids=source_ids,
                        status="FAIL",
                        support_score=0.0,
                        reason=f"引用了不存在的证据: {', '.join(unknown_sources)}",
                    )
                )
                continue

            clean_claim = self.remove_source_markers(claim)
            support_score = max(
                self.support_score(clean_claim, chunk_by_id[self._normalize_evidence_id(source_id)].text)
                for source_id in source_ids
            )

            if support_score >= self.min_support_score:
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        source_ids=source_ids,
                        status="PASS",
                        support_score=support_score,
                        reason="结论能被引用证据支持。",
                    )
                )
            else:
                unsupported_claims.append(claim)
                checks.append(
                    ClaimCheck(
                        claim=claim,
                        source_ids=source_ids,
                        status="FAIL",
                        support_score=support_score,
                        reason="引用证据不足以支持该结论。",
                    )
                )

        checked_claims = [
            check
            for check in checks
            if not self._is_uncertainty_statement(check.claim)
        ]
        passed_claims = sum(1 for check in checked_claims if check.status == "PASS")
        total_claims = len(checked_claims)
        overall_score = passed_claims / total_claims if total_claims else 1.0

        status = "PASS"
        if (
            missing_source_claims
            or unknown_source_claims
            or unsupported_claims
            or overall_score < self.min_overall_score
        ):
            status = "FAIL"

        return AccuracyReport(
            status=status,
            overall_score=overall_score,
            total_claims=total_claims,
            passed_claims=passed_claims,
            checks=checks,
            missing_source_claims=missing_source_claims,
            unknown_source_claims=unknown_source_claims,
            unsupported_claims=unsupported_claims,
        )

    def parse_evidence_chunks(self, rag_context: str) -> List[EvidenceChunk]:
        pattern = re.compile(r"【证据(\d+)】(?P<body>.*?)(?=【证据\d+】|\Z)", re.S)
        chunks: List[EvidenceChunk] = []

        for match in pattern.finditer(rag_context):
            evidence_id = f"证据{match.group(1)}"
            text = match.group("body").strip()

            if text:
                chunks.append(EvidenceChunk(evidence_id=evidence_id, text=text))

        return chunks

    def extract_claims(self, answer: str) -> List[str]:
        lines = []

        for raw_line in answer.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            line = re.sub(r"^[-*•\d.、\s]+", "", line).strip()

            if not line or line.endswith("：") or line.endswith(":"):
                continue

            if "[来源" in line:
                lines.append(line)
            else:
                parts = re.split(r"(?<=[。！？!?])\s*", line)
                lines.extend(part.strip() for part in parts if part.strip())

        claims = []

        for line in lines:
            if self._is_heading(line):
                continue

            claims.append(line)

        return claims

    def extract_source_ids(self, claim: str) -> List[str]:
        source_ids: List[str] = []

        for match in re.finditer(r"\[来源[:：]\s*([^\]]+)\]", claim):
            raw_sources = match.group(1)
            for item in re.split(r"[,，、;；]\s*", raw_sources):
                item = item.strip()
                if item:
                    source_ids.append(self._normalize_evidence_id(item))

        return source_ids

    def remove_source_markers(self, claim: str) -> str:
        return re.sub(r"\[来源[:：]\s*[^\]]+\]", "", claim).strip()

    def support_score(self, claim: str, evidence_text: str) -> float:
        claim_norm = self._normalize(claim)
        evidence_norm = self._normalize(evidence_text)

        if claim_norm and claim_norm in evidence_norm:
            return 1.0

        embedding_score = self._embedding_similarity(claim, evidence_text)
        keyword_score = self._keyword_coverage(claim, evidence_text)

        return max(0.0, min(1.0, embedding_score * 0.55 + keyword_score * 0.45))

    def _keyword_coverage(self, claim: str, evidence_text: str) -> float:
        claim_terms = [
            self._normalize(term)
            for term in self._extract_terms(claim)
            if len(self._normalize(term)) >= 2
        ]

        if not claim_terms:
            return 0.0

        evidence_norm = self._normalize(evidence_text)
        covered = sum(1 for term in claim_terms if term in evidence_norm)

        return covered / len(claim_terms)

    def _embedding_similarity(self, left: str, right: str) -> float:
        left_vector = self._hashed_embedding(left)
        right_vector = self._hashed_embedding(right)

        if not left_vector or not right_vector:
            return 0.0

        dot = 0.0
        for index, value in left_vector.items():
            dot += value * right_vector.get(index, 0.0)

        left_norm = math.sqrt(sum(value * value for value in left_vector.values()))
        right_norm = math.sqrt(sum(value * value for value in right_vector.values()))

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return max(0.0, min(1.0, dot / (left_norm * right_norm)))

    def _hashed_embedding(self, text: str) -> Dict[int, float]:
        vector: Dict[int, float] = {}

        for term in self._extract_terms(text):
            term_norm = self._normalize(term)
            if len(term_norm) < 2:
                continue

            digest = hashlib.md5(term_norm.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] = vector.get(index, 0.0) + sign

        return vector

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        terms: List[str] = []
        terms.extend(re.findall(r"[a-zA-Z0-9]+", text))

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_text = "".join(chinese_chars)

        for n in (2, 3, 4):
            for index in range(0, max(0, len(chinese_text) - n + 1)):
                terms.append(chinese_text[index:index + n])

        seen = set()
        unique_terms = []

        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    @staticmethod
    def _normalize(text: str) -> str:
        return (
            text.lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .replace("/", "")
            .strip()
        )

    @staticmethod
    def _normalize_evidence_id(source_id: str) -> str:
        source_id = source_id.strip()
        match = re.search(r"证据\s*(\d+)", source_id)

        if match:
            return f"证据{match.group(1)}"

        return source_id

    @staticmethod
    def _is_heading(text: str) -> bool:
        heading_text = text.strip("#：: ")
        return heading_text in {
            "基于证据的回答",
            "资料中未提到",
            "结论",
            "回答",
        }

    @staticmethod
    def _is_uncertainty_statement(text: str) -> bool:
        return any(
            keyword in text
            for keyword in (
                "资料中未提到",
                "证据不足",
                "无法确定",
                "不确定",
                "未提供",
            )
        )

    def _has_uncertainty_answer(self, answer: str) -> bool:
        return self._is_uncertainty_statement(answer)
