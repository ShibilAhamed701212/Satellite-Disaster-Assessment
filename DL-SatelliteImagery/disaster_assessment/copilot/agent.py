"""
Disaster Intelligence Copilot Agent.

Retrieval-Augmented Generation (RAG) agent that operates on actual
analysis results, GIS outputs, and user-provided documentation.

Status: IMPLEMENTED ARCHITECTURE
Requires external LLM provider for actual inference.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CopilotResponse:
    """Structured copilot response."""
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)
    status: str = "READY"


class CopilotAgent:
    """RAG-powered disaster intelligence copilot.

    Operates on:
    1. Actual analysis results from the pipeline
    2. GIS outputs and vector features
    3. Computed metrics and severity scores
    4. User-provided disaster documentation
    5. Indexed SOPs and guidelines

    Does NOT invent analysis results. All data comes through tools.
    """

    def __init__(self, provider: str = "ollama", model: str = "llama3"):
        self.provider = provider
        self.model = model
        self._context: Dict[str, Any] = {}
        self._tools: Dict[str, callable] = {}
        self._available = False

        self._register_default_tools()

    def _register_default_tools(self):
        """Register the default analysis tools."""
        self._tools = {
            "get_damage_statistics": self._get_damage_statistics,
            "get_flood_extent": self._get_flood_extent,
            "get_high_severity_regions": self._get_high_severity_regions,
            "get_vector_features": self._get_vector_features,
            "get_analysis_metadata": self._get_analysis_metadata,
        }

    def bind_analysis(self, report: Any):
        """Bind a DisasterReport to the copilot context."""
        self._context["report"] = report
        self._context["report_dict"] = report.to_dict() if hasattr(report, "to_dict") else {}

    def bind_vector_features(self, features: List[Any]):
        """Bind vector features to copilot context."""
        self._context["vector_features"] = features

    def register_tool(self, name: str, func: callable):
        """Register a custom tool."""
        self._tools[name] = func

    def query(self, question: str) -> CopilotResponse:
        """Query the copilot with a natural language question.

        Without an LLM provider, returns tool-based answers.
        With an LLM, would augment the question with RAG context.
        """
        # Try tool-based response first
        lower_q = question.lower()
        query_words = set(lower_q.split())

        # Find best matching tool (by number of overlapping keywords)
        best_tool = None
        best_overlap = 0
        for tool_name, tool_func in self._tools.items():
            tool_words = set(tool_name.replace("_", " ").split()) - {"get", "the"}
            overlap = len(tool_words & query_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_tool = (tool_name, tool_func)

        if best_tool and best_overlap > 0:
            tool_name, tool_func = best_tool
            try:
                result = tool_func()
                return CopilotResponse(
                    answer=str(result),
                    sources=[f"tool:{tool_name}"],
                    tool_calls=[tool_name],
                    status="COMPLETED",
                )
            except Exception:
                pass

        # Fallback: check if LLM is available
        if self._check_provider():
            return self._query_llm(question)

        return CopilotResponse(
            answer=(
                "The copilot requires an LLM provider to answer natural language questions. "
                f"Configured provider: {self.provider}. "
                "Please install and configure the provider, or use the analysis tools directly."
            ),
            status="PROVIDER_UNAVAILABLE",
        )

    def _check_provider(self) -> bool:
        """Check if the configured LLM provider is available."""
        if self.provider == "ollama":
            try:
                import requests
                resp = requests.get("http://localhost:11434/api/tags", timeout=3)
                return resp.status_code == 200
            except Exception:
                return False
        return False

    def _query_llm(self, question: str) -> CopilotResponse:
        """Query the LLM provider with RAG context."""
        try:
            import requests

            # Build context from analysis results
            context_parts = []
            report_dict = self._context.get("report_dict", {})
            if report_dict:
                context_parts.append(f"Analysis Results: {report_dict}")

            # Add vector feature summary
            features = self._context.get("vector_features", [])
            if features:
                context_parts.append(f"Vector Features Count: {len(features)}")

            context = "\n".join(context_parts) if context_parts else "No analysis data available."

            prompt = (
                f"You are a disaster intelligence assistant. "
                f"Based on the following analysis data:\n\n{context}\n\n"
                f"Answer the user's question: {question}"
            )

            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                return CopilotResponse(
                    answer=data.get("response", ""),
                    sources=["llm_context"],
                    status="COMPLETED",
                )
        except Exception:
            pass

        return CopilotResponse(
            answer="LLM query failed. Please check provider configuration.",
            status="FAILED",
        )

    # --- Built-in tools ---

    def _get_damage_statistics(self) -> Dict:
        report = self._context.get("report")
        if report is None:
            return {"error": "No analysis report bound"}
        d = report.to_dict()
        return d.get("building_damage", {})

    def _get_flood_extent(self) -> Dict:
        report = self._context.get("report")
        if report is None:
            return {"error": "No analysis report bound"}
        d = report.to_dict()
        return {
            "flood_percentage": d.get("flood_percentage", 0),
            "flood_area_m2": d.get("flood_area_m2"),
            "flood_area_km2": d.get("flood_area_km2"),
        }

    def _get_high_severity_regions(self) -> Dict:
        report = self._context.get("report")
        if report is None:
            return {"error": "No analysis report bound"}
        d = report.to_dict()
        return {
            "severity_score": d.get("severity_score", 0),
            "severity_level": d.get("severity_level", "UNKNOWN"),
        }

    def _get_vector_features(self) -> Dict:
        features = self._context.get("vector_features", [])
        return {"count": len(features), "type": "vector_features"}

    def _get_analysis_metadata(self) -> Dict:
        report = self._context.get("report")
        if report is None:
            return {"error": "No analysis report bound"}
        d = report.to_dict()
        return {
            "measurement_basis": d.get("measurement_basis", ""),
            "area_status": d.get("area_status", ""),
            "warnings": d.get("warnings", []),
        }
