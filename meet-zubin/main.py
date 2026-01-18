import os

from chat_node import ChatNode
from config import SYSTEM_PROMPT
from google import genai
from datetime import datetime

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.events import UserStartedSpeaking, UserStoppedSpeaking, UserTranscriptionReceived

GEMINI_API_KEY = os.getenv("ZP_GEMINI_API_KEY")
if GEMINI_API_KEY:
    # Separate clients for conversation vs search keeps configs isolated and code readable
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    search_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("No GEMINI_API_KEY found.")
    gemini_client = None
    search_client = None


async def call_handler(system: VoiceAgentSystem, call_request: CallRequest):

    today = datetime.now().strftime("%Y-%m-%d")

    # Main conversation node
    conversation_node = ChatNode(
        system_prompt=f"Today's date is ${today}. {SYSTEM_PROMPT}",
        gemini_client=gemini_client,
        search_client=search_client,
    )
    conversation_bridge = Bridge(conversation_node)
    system.with_speaking_node(conversation_node, bridge=conversation_bridge)

    conversation_bridge.on(UserTranscriptionReceived).map(
        conversation_node.add_event)

    (
        conversation_bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking, handler=conversation_node.on_interrupt_generate)
        .stream(conversation_node.generate)
        .broadcast()
    )

    await system.start()
    await system.send_initial_message(
        """Hello! I am Zubin's AI voice clone.
        I can answer questions about my work history and resumé.
        What would you like to know?"""
    )

    await system.wait_for_shutdown()


app = VoiceAgentApp(call_handler)

if __name__ == "__main__":
    app.run()
