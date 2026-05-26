import asyncio
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    JobProcess,
    StopResponse,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


def _format_search_results(payload: dict[str, Any], max_results: int) -> str:
    results = payload.get("results", [])
    if not results:
        return "No web results were found."

    formatted_results = []
    for result in results[:max_results]:
        title = result.get("title") or "Untitled result"
        url = result.get("url") or result.get("parsed_url") or ""
        snippet = result.get("content") or result.get("snippet") or ""

        parts = [title]
        if snippet:
            parts.append(snippet)
        if url:
            parts.append(f"Source: {url}")

        formatted_results.append("\n".join(parts))

    return "\n\n".join(formatted_results)


def _search_searxng(query: str, max_results: int) -> str:
    base_url = os.getenv("WEB_SEARCH_BASE_URL", "http://searxng:8080").rstrip("/")
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "safesearch": 1,
            "language": "en-US",
        }
    )
    url = f"{base_url}/search?{params}"

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "local-voice-ai/1.0"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return _format_search_results(payload, max_results)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_wake_words(value: Optional[str]) -> list[str]:
    if not value:
        return ["hey bella"]

    return [word.strip().lower() for word in value.split(",") if word.strip()]


def _strip_wake_word(text: str, wake_words: list[str]) -> tuple[bool, str]:
    for wake_word in wake_words:
        wake_word_pattern = r"[\s,!.?]+".join(
            re.escape(part) for part in wake_word.split()
        )
        pattern = r"^\s*" + wake_word_pattern + r"\b[\s,!.?]*"
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return True, text[match.end() :].strip()

    return False, text


def _is_sleep_request(text: str) -> bool:
    normalized_text = text.lower().strip(" .!?")
    return normalized_text in {
        "go to sleep",
        "stop listening",
        "sleep",
        "standby",
        "stand by",
    }


class Assistant(Agent):
    def __init__(
        self,
        wake_word_enabled: bool = True,
        wake_words: Optional[list[str]] = None,
    ) -> None:
        self.wake_word_enabled = wake_word_enabled
        self.wake_words = wake_words or ["hey bella"]
        self.awake = not wake_word_enabled
        wake_word_instructions = ""
        if wake_word_enabled:
            wake_word_instructions = f"""
            Wake word mode is enabled. The configured wake phrases are: {", ".join(self.wake_words)}.
            Only respond after the user says a configured wake phrase, or after you have already been woken up.
            If a user message does not contain a configured wake phrase while you are asleep, produce no response.
            If the user asks you to sleep or stop listening, briefly acknowledge and then wait for the wake phrase again."""

        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Use the search_web tool when the user asks you to search, look something up, find current information, or answer questions that may depend on recent events.
            When you use web results, summarize them conversationally and mention the source names or websites when useful.
            {wake_word_instructions}
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""".format(
                wake_word_instructions=wake_word_instructions
            ),
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        if not self.wake_word_enabled:
            return

        text = new_message.text_content or ""
        if _is_sleep_request(text):
            self.awake = False
            new_message.content = [
                "The user asked you to go to sleep. Briefly acknowledge that you will wait for the wake word."
            ]
            return

        heard_wake_word, command = _strip_wake_word(text, self.wake_words)
        if heard_wake_word:
            self.awake = True
            if command:
                new_message.content = [command]
            else:
                new_message.content = [
                    "The user woke you up. Briefly acknowledge and ask how you can help."
                ]
            return

        if not self.awake:
            logger.info("Ignoring user turn while waiting for wake word: %s", text)
            raise StopResponse()

    @function_tool()
    async def multiply_numbers(
        self,
        context: RunContext,
        number1: int,
        number2: int,
    ) -> dict[str, Any]:
        """Multiply two numbers.
        
        Args:
            number1: The first number to multiply.
            number2: The second number to multiply.
        """

        return f"The product of {number1} and {number2} is {number1 * number2}."

    @function_tool()
    async def search_web(
        self,
        context: RunContext,
        query: str,
        max_results: int = 3,
    ) -> str:
        """Search the web for current information.

        Args:
            query: The web search query.
            max_results: The maximum number of search results to return.
        """

        max_results = max(1, min(max_results, 5))
        try:
            return await asyncio.to_thread(_search_searxng, query, max_results)
        except Exception as exc:
            logger.exception("Web search failed")
            return f"I could not complete the web search because {exc}."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    llama_model = os.getenv("LLAMA_MODEL", "qwen3-4b")
    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://llama_cpp:11434/v1")

    stt_provider = os.getenv("STT_PROVIDER", "nemotron").lower()
    if stt_provider == "whisper":
        default_stt_base_url = "http://whisper:80/v1"
        default_stt_model = "Systran/faster-whisper-small"
    else:
        default_stt_base_url = "http://nemotron:8000/v1"
        default_stt_model = "nemotron-speech-streaming"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model = os.getenv("STT_MODEL", default_stt_model)
    stt_api_key = os.getenv("STT_API_KEY", "no-key-needed")
    tts_voice = os.getenv("TTS_VOICE", "af_bella")
    wake_word_enabled = _env_bool("WAKE_WORD_ENABLED", default=True)
    wake_words = _parse_wake_words(os.getenv("WAKE_WORDS"))

    logger.info(
        "Starting agent with STT provider=%s model=%s base_url=%s",
        stt_provider,
        stt_model,
        stt_base_url,
    )
    logger.info(
        "Wake word enabled=%s wake_words=%s",
        wake_word_enabled,
        wake_words,
    )

    session = AgentSession(
        stt=openai.STT(
            base_url=stt_base_url,
            # base_url="http://localhost:11435/v1", # uncomment for local testing
            model=stt_model,
            api_key=stt_api_key
        ),
        llm=openai.LLM(
            base_url=llama_base_url,
            # base_url="http://localhost:11436/v1", # uncomment for local testing
            model=llama_model,
            api_key="no-key-needed"
        ),
        tts=openai.TTS(
            base_url="http://kokoro:8880/v1",
            # base_url="http://localhost:8880/v1", # uncomment for local testing
            model="kokoro",
            voice=tts_voice,
            api_key="no-key-needed"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=not wake_word_enabled,
    )

    await session.start(
        agent=Assistant(
            wake_word_enabled=wake_word_enabled,
            wake_words=wake_words,
        ),
        room=ctx.room,
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)
