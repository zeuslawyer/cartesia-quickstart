"""
Tools for the voice agent - SearchWebTool for Gemini grounded search
"""

from google.genai import types as gemini_types


class SearchWebTool:
    """Tool for searching the web using Gemini's Google Search grounding"""

    @staticmethod
    def name() -> str:
        return "search_web"

    @staticmethod
    def to_gemini_tool() -> gemini_types.FunctionDeclaration:
        return gemini_types.FunctionDeclaration(
            name=SearchWebTool.name(),
            description=(
                "ALWAYS use this tool first when the user asks about recent events, "
                "publications, podcasts, videos, talks, or activities after August 2025. "
                "Do not apologize or end the call - search instead."
            ),
            parameters=gemini_types.Schema(
                type=gemini_types.Type.OBJECT,
                properties={
                    "query": gemini_types.Schema(
                        type=gemini_types.Type.STRING,
                        description="The user's question needing recent information",
                    ),
                },
                required=["query"],
            ),
        )
