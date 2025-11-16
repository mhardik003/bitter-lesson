import os
from typing import List, Dict, Any, Optional, TypedDict

import time
import streamlit as st
import dspy


# ============================================================
# 1. Types for storing traces in Streamlit session_state
# ============================================================


class TraceData(TypedDict):
    """Minimal structured representation of what we want to replay later."""

    thinking: str  # final "next_thought" text
    tools: List[str]  # list of markdown lines like "- Calling tool ..."


class ChatMessage(TypedDict, total=False):
    """Chat history entry.

    role: "user" or "assistant"
    content: rendered message text
    trace: optional TraceData for assistant messages that came from the agent
    """

    role: str
    content: str
    trace: TraceData


# ============================================================
# 2. DSPy configuration: LM and ReAct agent
# ============================================================

# Configure DSPy LM.
# cache=False so that repeated questions still stream instead of returning from cache.
lm = dspy.LM("groq/openai/gpt-oss-20b")
dspy.configure(lm=lm)


# Demo tools so you can see tool calls in the UI.
# Replace these with your own tools later.
def get_weather(city: str) -> str:
    """Fake weather tool to demonstrate tool calls."""
    time.sleep(5)
    return f"The weather in {city} is mildly chaotic with 37 percent chance of bugs."


def search_web(query: str) -> str:
    """Fake search tool to demonstrate tool calls."""
    time.sleep(5)
    return f"Search results for '{query}': imagine a list of useful links."


# Basic ReAct agent.
# You can later replace signature="question->answer" with your own Signature class.
react_agent = dspy.ReAct(
    signature="question->answer",
    tools=[get_weather, search_web],
    max_iters=6,
)


# ============================================================
# 3. Streaming helpers: Status messages and StreamListener
# ============================================================


class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    """Customize which status messages we stream.

    We only care about tool-level hooks here, because that is what the UI
    will display in the right column.
    """

    def tool_start_status_message(self, instance: Any, inputs: Dict[str, Any]) -> str:
        # instance is a dspy.Tool, has a .name field
        return f"Calling tool `{instance.name}` with inputs {inputs}..."

    def tool_end_status_message(self, outputs: Any) -> str:
        # Outputs are whatever your tool returns; we keep it short here.
        return f"Tool finished with output: {str(outputs)[:200]}..."


# Listen to the ReAct field "next_thought", which ReAct uses for the reasoning string.
# allow_reuse=True is important because ReAct emits next_thought multiple times
# across its internal tool loop.
stream_listeners = [
    dspy.streaming.StreamListener(
        signature_field_name="next_thought",
        allow_reuse=True,
    )
]

# Wrap the ReAct agent with streaming.
# async_streaming=False gives us a plain Python generator instead of an async one,
# which is simpler to integrate with Streamlit.
stream_react = dspy.streamify(
    react_agent,
    status_message_provider=MyStatusMessageProvider(),
    stream_listeners=stream_listeners,
    async_streaming=False,
)


# ============================================================
# 4. Streamlit layout helpers
# ============================================================


def render_trace_static(trace: TraceData) -> None:
    """Render an existing trace (for old messages) in the standard layout.

    This uses the same structure as the live streaming view:
    - expander
    - two columns: thinking on the left, tools on the right
    """
    with st.expander("Agent thinking and tools", expanded=False):
        col_thinking, col_tools = st.columns(2)

        with col_thinking:
            thinking_text = trace["thinking"] or "(no thinking streamed)"
            st.markdown(f"**Thinking:** {thinking_text}")

        with col_tools:
            tool_lines = trace["tools"]
            if tool_lines:
                st.markdown("\n".join(tool_lines))
            else:
                st.markdown("_No tools were called._")


def run_react_with_stream(question: str) -> (Optional[dspy.Prediction], TraceData):
    """Run the streamified ReAct agent and build up a TraceData structure.

    This function is pure DSPy and Python. It does not touch Streamlit directly.
    The caller is responsible for wiring its updates into the UI.

    We stream:
    - StreamResponse chunks for "next_thought" tokens
    - StatusMessage chunks for tool start/end

    At the end we return:
    - final dspy.Prediction (or None)
    - final TraceData, which you can both display and store in session_state
    """
    output_stream = stream_react(question=question)

    final_pred: Optional[dspy.Prediction] = None
    accumulated_thought = ""
    tool_lines: List[str] = []

    for chunk in output_stream:
        # 1) Token streaming for the "next_thought" field
        if isinstance(chunk, dspy.streaming.StreamResponse):
            if chunk.signature_field_name == "next_thought":
                accumulated_thought += chunk.chunk

        # 2) Status messages from StatusMessageProvider (we configured tool hooks)
        elif isinstance(chunk, dspy.streaming.StatusMessage):
            tool_lines.append(f"- {chunk.message}")

        # 3) Final program output
        elif isinstance(chunk, dspy.Prediction):
            final_pred = chunk

    trace: TraceData = {
        "thinking": accumulated_thought,
        "tools": tool_lines,
    }
    return final_pred, trace


def stream_react_into_ui(question: str) -> (Optional[dspy.Prediction], TraceData):
    """Run the agent and stream updates into the live UI.

    This function bridges DSPy streaming into Streamlit:

    - Creates the expander and two columns
    - Uses placeholders so we can update thinking and tools while the stream runs
    - Calls run_react_with_stream under the hood, but replays the same logic
      while updating the placeholders in real time.
    """
    # Live expander for this turn
    with st.expander("Agent thinking and tools", expanded=True):
        col_thinking, col_tools = st.columns(2)

        with col_thinking:
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("**Thinking:** ")

        with col_tools:
            tools_placeholder = st.empty()
            tools_placeholder.markdown("_No tools yet..._")

    # Now run the streaming loop, but update UI as we go.
    output_stream = stream_react(question=question)

    final_pred: Optional[dspy.Prediction] = None
    accumulated_thought = ""
    tool_lines: List[str] = []

    for chunk in output_stream:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            if chunk.signature_field_name == "next_thought":
                accumulated_thought += chunk.chunk
                # Update left column thinking text
                thinking_placeholder.markdown(f"**Thinking:** {accumulated_thought}")

        elif isinstance(chunk, dspy.streaming.StatusMessage):
            tool_lines.append(f"- {chunk.message}")
            # Update right column tool log
            tools_placeholder.markdown("\n".join(tool_lines))

        elif isinstance(chunk, dspy.Prediction):
            final_pred = chunk

    trace: TraceData = {
        "thinking": accumulated_thought,
        "tools": tool_lines,
    }
    return final_pred, trace


# ============================================================
# 5. Streamlit app: chat loop
# ============================================================

st.set_page_config(page_title="DSPy ReAct + Streamlit", page_icon="🤖")

st.title("DSPy ReAct Agent with Streaming Tools")
st.write(
    "Ask a question. The ReAct agent will think step by step and call tools. "
    "You see its thinking and tool calls live, and past traces stay attached to each reply."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages: List[ChatMessage] = []  # type: ignore[assignment]


# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # For assistant messages that have a stored trace, replay it
        if msg.get("role") == "assistant" and "trace" in msg:
            trace_data = msg["trace"]  # type: ignore[index]
            render_trace_static(trace_data)

# New user input
user_query = st.chat_input(
    "Ask something, for example: "
    "`What is the weather in the city where Alan Turing was born?`"
)

if user_query:
    # 1) Show and store user message
    st.session_state.messages.append(ChatMessage(role="user", content=user_query))

    with st.chat_message("user"):
        st.markdown(user_query)

    # 2) Agent reply with live streaming
    with st.chat_message("assistant"):
        # Stream reasoning + tools into the UI
        pred, trace = stream_react_into_ui(user_query)

        # Extract final natural language answer from the Prediction
        if pred is not None and hasattr(pred, "answer"):
            answer_text = str(pred.answer)
        elif pred is not None:
            # If your signature uses a different output field, adjust here.
            answer_text = str(pred)
        else:
            answer_text = "Agent did not return an answer."

        st.markdown(f"**Answer:** {answer_text}")

    # 3) Store assistant message along with trace so style persists on rerun
    st.session_state.messages.append(
        ChatMessage(
            role="assistant",
            content=answer_text,
            trace=trace,
        )
    )
