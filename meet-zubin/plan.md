# Plan: Add Gemini Search Grounding to Voice Agent

## Goal

Supplement the voice agent's context with live web search using Gemini's native Google Search grounding (not SERP API).

---

## Why Two Gemini Clients?

- **Main client**: Handles conversation with `EndCallTool`, no search grounding
- **Search client**: Handles web search queries with `googleSearch` tool enabled
- Keeps concerns separated; avoids search grounding interfering with normal conversation flow

---

## Files to Modify

| File             | Changes                                                   |
| ---------------- | --------------------------------------------------------- |
| `main.py`        | Create second `genai.Client` for search                   |
| `chat_node.py`   | Add search tool, handle function call, call search client |
| `tools.py` (new) | Define `SearchWebTool` + `SearchWebArgs`                  |

---

## Implementation Steps

### 1. Create `tools.py`

- Define `SearchWebArgs(BaseModel)` with `query: str` field
- Define `SearchWebTool` class with:
  - `name()` -> `"search_web"`
  - `to_gemini_tool()` -> returns Gemini `FunctionDeclaration`

### 2. Update `main.py`

- Create two Gemini clients (can share same API key):
  ```python
  gemini_client = genai.Client(api_key=GEMINI_API_KEY)        # conversation
  search_client = genai.Client(api_key=GEMINI_API_KEY)        # web search
  ```
- Pass both to `ChatNode`:
  ```python
  ChatNode(
      system_prompt=SYSTEM_PROMPT,
      gemini_client=gemini_client,
      search_client=search_client,
  )
  ```

### 3. Update `chat_node.py`

**`__init__`:**

- Accept `search_client` parameter
- Add `SearchWebTool.to_gemini_tool()` to `generation_config.tools`
- Create search-specific config with grounding:
  ```python
  self.search_config = GenerateContentConfig(
      tools=[{"google_search": {}}],
  )
  ```

**`process_context`:**

- Detect `function_call.name == "search_web"`
- Extract query from args
- Call search client with grounding:
  ```python
  search_response = await self.search_client.aio.models.generate_content(
      model="gemini-2.5-flash",
      contents=query,
      config=self.search_config,
  )
  ```
- Yield `AgentResponse` with summarized search results
- Optionally log grounding metadata (sources)

---

## Flow Diagram

```
User asks about Zubin's background
        |
        v
ChatNode detects need for more info
        |
        v
LLM calls search_web tool with query
        |
        v
ChatNode intercepts function call
        |
        v
Search client queries Gemini + Google Search grounding
        |
        v
Results returned, injected as AgentResponse
        |
        v
Agent speaks the supplemented answer
```

---

## Considerations

- **Latency**: Search adds round-trip time; may want to keep queries brief
- **Token usage**: Search responses can be verbose; summarize before speaking
- **Prompt guidance**: Update `SYSTEM_PROMPT` to instruct when to use search tool
- **Error handling**: Handle search failures gracefully (fallback to existing knowledge)

---

## Testing

- Verify normal conversation still works without triggering search
- Test queries that should trigger search (e.g., "What's Zubin's latest work?")
- Confirm grounding metadata is accessible for source attribution
