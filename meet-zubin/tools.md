# Tool Calling in Cartesia Line SDK

## Concept
Tools = functions LLM can call mid-conversation. LLM decides when to use them.

**Flow:** User speaks → LLM triggers tool → Tool runs → Result back to LLM → LLM speaks response

---

## 3 Steps to Add Tools

### 1. Define tool (Pydantic model):
```python
class EndCallTool(BaseModel):
    @staticmethod
    def to_gemini_tool():
        return Tool(function_declarations=[
            FunctionDeclaration(
                name="end_call",
                description="End the call",
                parameters=Schema(type=Type.OBJECT, properties={})
            )
        ])
```

### 2. Register with LLM config:
```python
self.generation_config = GenerateContentConfig(
    tools=[EndCallTool.to_gemini_tool()],
)
```

### 3. Handle in `process_context()`:
```python
for part in msg.candidates[0].content.parts:
    if part.function_call:
        if part.function_call.name == "end_call":
            yield EndCall()  # System receives this
            return
```

---

## Architecture Concepts

| Component | Role |
|-----------|------|
| **Node** | Contains LLM logic, yields events |
| **Bridge** | Routes events between nodes & system |
| **System** | Manages call, sends audio to Cartesia |

### Bridge pattern:
```python
bridge.on(UserStoppedSpeaking).stream(node.generate).broadcast()
```
Meaning: When user stops talking → run node's generate() → send yielded events to system

### Key events:
- `AgentResponse` → text to speak
- `EndCall` → hang up
- `ToolCall`/`ToolResult` → tool execution cycle

---

## Full Tool Definition Pattern

```python
from pydantic import BaseModel
from google.genai import Tool, FunctionDeclaration, Schema, Type

class CustomTool(BaseModel):
    """Your tool description."""
    param1: str
    param2: int = 0

    @staticmethod
    def to_gemini_tool():
        return Tool(function_declarations=[
            FunctionDeclaration(
                name="custom_tool_name",
                description="What this tool does",
                parameters=Schema(
                    type=Type.OBJECT,
                    properties={
                        "param1": Schema(type=Type.STRING, description="..."),
                        "param2": Schema(type=Type.INTEGER, description="..."),
                    },
                    required=["param1"]
                )
            )
        ])
```

---

## Event Flow

1. `UserStoppedSpeaking` triggers `generate()`
2. Node's `process_context()` calls LLM
3. LLM returns `function_call` in response
4. Node detects `function_call` and executes tool logic
5. Node yields `ToolCall` event (or `EndCall`, `AgentResponse`)
6. Bridge's `.broadcast()` sends events to VoiceAgentSystem
7. System processes event (speaks response, ends call, etc.)

---

## Key Points

- Tools = Pydantic models with conversion methods (`to_gemini_tool()`, `to_openai_tool()`)
- Register in generation config
- Check `function_call` in LLM response parts
- Yield appropriate events (`EndCall`, `ToolCall`, `AgentResponse`)
- Bridge's `.broadcast()` sends events to system
