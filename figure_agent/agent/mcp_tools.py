import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import httpx

from figure_agent.agent.dynamic_planner import AgentToolSpec


@dataclass
class MCPToolDefinition:
    name: str
    description: str
    endpoint: str = ""
    server: str = ""

    @property
    def planner_name(self) -> str:
        return self.name if self.name.startswith("mcp:") else f"mcp:{self.name}"


class MCPToolRegistry:
    """
    Lightweight adapter for exposing MCP Server tools to the Agent planner.

    The project can point MCP_TOOL_MANIFEST_PATH to a JSON file:
    {
      "tools": [
        {
          "name": "paper_meta_search",
          "description": "Search external paper metadata",
          "endpoint": "http://127.0.0.1:8765/tools/paper_meta_search"
        }
      ]
    }

    The endpoint is expected to accept JSON {"query": "...", "context": {...}}
    and return JSON with "result" or "content". This keeps the Agent side
    decoupled from a specific MCP transport while still allowing MCP tool
    surfaces to participate in planning and execution.
    """

    def __init__(self, tools: List[MCPToolDefinition] | None = None):
        self.tools: Dict[str, MCPToolDefinition] = {
            tool.planner_name: tool
            for tool in tools or []
        }

    @classmethod
    def from_env(cls) -> "MCPToolRegistry":
        manifest_path = os.environ.get("MCP_TOOL_MANIFEST_PATH", "").strip()
        if not manifest_path:
            return cls()

        path = Path(manifest_path)
        if not path.exists():
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        tools = []
        for item in data.get("tools", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            if not name or not description:
                continue
            tools.append(
                MCPToolDefinition(
                    name=name,
                    description=description,
                    endpoint=str(item.get("endpoint", "")).strip(),
                    server=str(item.get("server", "")).strip(),
                )
            )

        return cls(tools)

    def list_tool_specs(self) -> List[AgentToolSpec]:
        return [
            AgentToolSpec(
                name=tool.planner_name,
                description=f"MCP Server 工具：{tool.description}",
            )
            for tool in self.tools.values()
        ]

    def call_tool(self, tool_name: str, query: str, context: Dict | None = None) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            return f"MCP 工具未注册：{tool_name}"

        if not tool.endpoint:
            return f"MCP 工具 {tool_name} 已注册，但没有配置 endpoint"

        try:
            response = httpx.post(
                tool.endpoint,
                json={
                    "query": query,
                    "context": context or {},
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return f"MCP 工具 {tool_name} 调用失败：{exc}"

        result = data.get("result", data.get("content", data))
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
