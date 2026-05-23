# figure_agent/agent/skill_loader.py

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SkillConfig:
    """
    单个 skill 的配置。
    """
    name: str
    file_name: str
    description: str
    keywords: Tuple[str, ...]
    patterns: Tuple[str, ...] = ()


class SkillLoader:
    """
    Skill 加载器。

    职责：
    1. 根据用户输入判断是否需要加载某个 skill
    2. 从 Skill 文件夹中读取对应 skill.md
    3. 将 skill 内容包装成 system message
    4. 每轮按需注入，不长期保存到 memory，避免 token 污染

    当前版本：
    - 支持关键词匹配
    - 支持正则匹配
    """

    DEFAULT_SKILLS: Dict[str, SkillConfig] = {
        "paper_search": SkillConfig(
            name="paper_search",
            file_name="paper_search.md",
            description="论文搜索 skill，用于查找和整理相关论文。",
            keywords=(
                "搜索论文",
                "查找论文",
                "推荐论文",
            ),
            patterns=(
                r"找.*论文",
                r"搜索.*论文",
                r"查找.*论文",
                r"推荐.*论文",
                r"整理.*论文",
            ),
        ),

        "paper_reading": SkillConfig(
            name="paper_reading",
            file_name="paper_reading.md",
            description="论文阅读 skill，用于总结论文、提取创新点、分析方法和实验。",
            keywords=(
                "阅读论文",
                "总结论文",
                "分析论文",
                "分析提取论文创新点",
                "提取论文创新点",
            ),
            patterns=(
                r"读.*论文",
                r"阅读.*论文",
                r"总结.*论文",
                r"分析.*论文",
                r"提取.*创新点",
            ),
        ),

        "model_figure": SkillConfig(
            name="model_figure",
            file_name="model_figure.md",
            description="模型图生成 skill，用于生成流程图、结构图和文生图 Prompt。",
            keywords=(
                "生成模型图",
                "生成流程图",
                "生成结构图",
                "画模型结构图",
            ),
            patterns=(
                r"生成.*模型图",
                r"生成.*流程图",
                r"生成.*结构图",
                r"画.*模型图",
                r"画.*流程图",
            ),
        ),

        "research_innovation": SkillConfig(
            name="research_innovation",
            file_name="research_innovation.md",
            description="科研创新点分析 skill，用于提炼、设计和优化科研创新点。",
            keywords=(
                "如何创新性",
                "怎么科研创新",
                "方法创新",
                "设计改进点",
            ),
            patterns=(
                r".*如何.*创新.*",
                r".*怎么.*创新.*",
                r".*设计.*创新点.*",
            ),
        ),
    }

    def __init__(
        self,
        skill_dir: str = "Skill",
        skills: Optional[Dict[str, SkillConfig]] = None,
        cache_enabled: bool = True,
    ):
        self.skill_dir = Path(skill_dir)

        if not self.skill_dir.is_absolute():
            self.skill_dir = Path.cwd() / self.skill_dir

        self.skills = skills or self.DEFAULT_SKILLS
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, str] = {}

    def detect_skills(self, user_input: str) -> List[str]:
        """
        根据用户输入判断需要加载哪些 skill。

        匹配方式：
        1. keywords：普通关键词子串匹配
        2. patterns：正则表达式匹配

        示例：
        - “帮我生成 Transformer 的模型图”
          会命中 r"生成.*模型图"

        - “帮我画一个 CNN 的流程图”
          会命中 r"画.*流程图"
        """
        text = user_input.lower()
        matched_skills: List[str] = []

        for skill_name, config in self.skills.items():
            keyword_hit = self._match_keywords(
                text=text,
                keywords=config.keywords,
            )

            pattern_hit = self._match_patterns(
                text=user_input,
                patterns=config.patterns,
            )

            if keyword_hit or pattern_hit:
                matched_skills.append(skill_name)

        return self._sort_skills(matched_skills)

    def load_skill(self, skill_name: str) -> str:
        """
        读取指定 skill 的 markdown 内容。
        """
        if skill_name not in self.skills:
            raise ValueError(f"未知 skill: {skill_name}")

        if self.cache_enabled and skill_name in self._cache:
            return self._cache[skill_name]

        config = self.skills[skill_name]
        skill_path = self.skill_dir / config.file_name

        if not skill_path.exists():
            raise FileNotFoundError(
                f"未找到 skill 文件: {skill_path}"
            )

        content = skill_path.read_text(encoding="utf-8").strip()

        if not content:
            raise ValueError(f"skill 文件为空: {skill_path}")

        if self.cache_enabled:
            self._cache[skill_name] = content

        return content

    def build_skill_messages(self, user_input: str) -> List[Dict[str, str]]:
        """
        根据用户输入构建本轮需要注入的 skill system messages。

        注意：
        skill 内容只在本轮注入，不应长期写入 memory。
        """
        skill_names = self.detect_skills(user_input)
        skill_messages: List[Dict[str, str]] = []

        for skill_name in skill_names:
            config = self.skills[skill_name]

            try:
                skill_content = self.load_skill(skill_name)

                message = {
                    "role": "system",
                    "content": (
                        f"本轮用户问题触发了 skill：{config.name}。\n"
                        f"skill 说明：{config.description}\n\n"
                        f"请严格遵循以下 skill.md 中的规则执行任务：\n\n"
                        f"{skill_content}"
                    ),
                }

            except FileNotFoundError as exc:
                message = {
                    "role": "system",
                    "content": (
                        f"本轮用户问题可能需要 skill：{config.name}，"
                        f"但没有找到对应的 skill.md 文件。\n"
                        f"缺失文件：{exc}\n"
                        f"请在回答中提醒用户或上层程序补充该 skill 文件。"
                    ),
                }

            except ValueError as exc:
                message = {
                    "role": "system",
                    "content": (
                        f"本轮用户问题可能需要 skill：{config.name}，"
                        f"但 skill 文件存在问题：{exc}\n"
                        f"请在回答中提醒用户检查该 skill 文件。"
                    ),
                }

            skill_messages.append(message)

        return skill_messages

    def available_skills(self) -> List[str]:
        """
        返回当前支持的 skill 名称。
        """
        return list(self.skills.keys())

    def clear_cache(self) -> None:
        """
        清空 skill 文件缓存。
        """
        self._cache.clear()

    @staticmethod
    def _match_keywords(
        text: str,
        keywords: Tuple[str, ...],
    ) -> bool:
        """
        关键词匹配。

        text 传入前已经 lower。
        """
        for keyword in keywords:
            if keyword.lower() in text:
                return True

        return False

    @staticmethod
    def _match_patterns(
        text: str,
        patterns: Tuple[str, ...],
    ) -> bool:
        """
        正则匹配。
        """
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True

        return False

    @staticmethod
    def _sort_skills(skill_names: List[str]) -> List[str]:
        """
        固定 skill 执行顺序。

        例如用户说：
        “帮我找论文，读一下，并生成模型图”

        应该按照：
        paper_search -> paper_reading -> model_figure
        """
        order = {
            "paper_search": 0,
            "paper_reading": 1,
            "model_figure": 2,
        }

        return sorted(
            skill_names,
            key=lambda name: order.get(name, 999),
        )