"""
GeminiReasoningNode - Voice-optimized ReasoningNode implementation using proven Gemini logic
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional, Union

from config import DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE
from google import genai
from google.genai import types as gemini_types
from loguru import logger

from line.events import AgentResponse, EndCall
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode
from line.tools.system_tools import EndCallArgs, EndCallTool, end_call
from line.utils.gemini_utils import convert_messages_to_gemini
from tools import SearchWebTool


class ChatNode(ReasoningNode):
    """ 
    Voice-optimized ReasoningNode using template method pattern with Gemini streaming.
    - Uses ReasoningNode's template method generate() for consistent flow
    - Implements process_context() for Gemini streaming
    - Integrates with end_call tool
    """

    def __init__(
        self,
        system_prompt: str,
        gemini_client: Optional[genai.Client] = None,
        search_client: Optional[genai.Client] = None,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        max_context_length: int = 100,
        max_output_tokens: int = 350,
    ):
        """
        Initialize the Voice reasoning node with proven Gemini configuration

        Args:
            system_prompt: System prompt for the LLM
            gemini_client: Google Gemini client instance.
                If not provided, a canned (dummy) response will be streamed.
            search_client: Separate Gemini client for web search with grounding.
            model_id: Gemini model ID to use
            temperature: Temperature for generation
            max_context_length: Maximum number of conversation turns to keep
        """
        super().__init__(system_prompt=system_prompt, max_context_length=max_context_length)

        self.client = gemini_client
        self.search_client = search_client
        self.model_id = model_id
        self.temperature = temperature

        # Interruption support
        self.stop_generation_event = None

        # Create generation config - includes search_web tool so LLM can request searches
        self.generation_config = gemini_types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            tools=[EndCallTool.to_gemini_tool(), SearchWebTool.to_gemini_tool()],
            max_output_tokens=max_output_tokens,
            thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
        )

        # Search config uses Google Search grounding - separate from main config
        # to avoid grounding interfering with normal conversation flow
        self.search_config = gemini_types.GenerateContentConfig(
            tools=[gemini_types.Tool(
                google_search=gemini_types.GoogleSearch())],
        )

        logger.info(
            f"[Chat_Node]: GeminiNode initialized with model: {model_id}")

    async def process_context(
        self, context: ConversationContext
    ) -> AsyncGenerator[Union[AgentResponse, EndCall], None]:
        """
        Process the conversation context and yield responses from Gemini.

        Yields:
            AgentResponse: Text chunks from Gemini
            AgentEndCall: end_call Event
        """
        if not context.events:
            logger.info(" [Chat_Node]: No messages to process")
            return

        messages = convert_messages_to_gemini(context.events)

        user_message = context.get_latest_user_transcript_message()
        if user_message:
            logger.info(f'🧠 Processing user message: "{user_message}"')

        full_response = ""
        if not self.client:
            stream = canned_gemini_response_stream()
        else:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model_id,
                contents=messages,
                config=self.generation_config,
            )

        async for msg in stream:
            if msg.text:
                full_response += msg.text
                yield AgentResponse(content=msg.text)

            if msg.function_calls:
                for function_call in msg.function_calls:
                    if function_call.name == EndCallTool.name():
                        goodbye_message = function_call.args.get(
                            "goodbye_message", "Goodbye!")
                        args = EndCallArgs(goodbye_message=goodbye_message)
                        logger.info(
                            f"🤖 End call tool called. Ending conversation with goodbye message: "
                            f"{args.goodbye_message}"
                        )
                        async for item in end_call(args):
                            yield item

                    elif function_call.name == SearchWebTool.name():
                        user_query = function_call.args.get("query", "")
                        async for item in self._execute_grounded_search(user_query):
                            yield item

        if full_response:
            logger.info(
                f'🤖 Agent response: "{full_response}" ({len(full_response)} chars)')

    async def _execute_grounded_search(
        self, user_query: str
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Execute a grounded web search and yield the response.

        We build the search query here rather than asking the LLM to generate it,
        avoiding an extra round-trip that would add latency to voice responses.
        """
        if not self.search_client:
            logger.warning("Search client not available, skipping search")
            yield AgentResponse(content="I'm unable to search for recent information right now.")
            return

        # Build query with date filter - 6 months ago ensures we get recent info only
        six_months_ago = (datetime.now() - timedelta(days=180)
                          ).strftime("%Y-%m-%d")
        search_query = f"\"Zubin Pratap\" + {user_query} + after:{six_months_ago}"

        logger.info(
            f"[Chat_Node]: 🔍 Executing grounded search: {search_query}")

        try:
            response = await self.search_client.aio.models.generate_content(
                model=self.model_id,
                contents=search_query,
                config=self.search_config,
            )

            if response.text:
                logger.info(
                    f"[Chat_Node]: 🔍 Search result (truncated): {response.text[:200]}...")
                yield AgentResponse(content=response.text)

            # Log grounding sources for debugging/attribution
            metadata = response.candidates[0].grounding_metadata if response.candidates else None
            if metadata and metadata.grounding_chunks:
                sources = [
                    chunk.web.uri for chunk in metadata.grounding_chunks if chunk.web]
                logger.info(
                    f"[Chat_Node]: 🔍 {len(sources) }Grounding sources: {sources}")

        except Exception as e:
            logger.error(f"Search failed: {e}")
            yield AgentResponse(content="I couldn't find recent information on that topic.")


async def canned_gemini_response_stream() -> AsyncGenerator[gemini_types.GenerateContentResponse, None]:
    """
    Stream a canned response from Gemini.

    This is to support running this example without a Gemini API key.
    """
    # Random messages about missing API key
    api_key_messages = [
        "I am a silly AI assistant because you didn't provide a Gemini API key. Add it to your environment variables.",
        "My brain is offline because I am missing a Gemini API key! Add the key to your environment variables.",
        "I'm like a car without keys - can't go anywhere. Add your Gemini API key for intelligence.",
    ]

    # Select a random message
    message = random.choice(api_key_messages)

    # Create the response structure
    part = gemini_types.Part(text=message)
    content = gemini_types.Content(parts=[part], role="model")
    candidate = gemini_types.Candidate(content=content, finish_reason="STOP")
    response = gemini_types.GenerateContentResponse(candidates=[candidate])

    await asyncio.sleep(0.005)
    yield response
