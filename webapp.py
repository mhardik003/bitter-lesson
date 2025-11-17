import os
from typing import List, Dict, Any, Optional, TypedDict, Tuple

import streamlit as st
import dspy


# import logging
# logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

from agents.search_agent import create_case_study_agent, create_chat_agent  # your file


# ===============================
# 1. Types for stored traces
# ===============================


class ToolTrace(TypedDict):
    name: str
    inputs: str
    outputs: str


class TraceData(TypedDict):
    thinking: str  # final "next_thought" text
    tools: List[ToolTrace]  # each tool call with full inputs and outputs


class ChatMessage(TypedDict, total=False):
    role: str  # "user" or "assistant"
    content: str
    trace: TraceData  # only for assistant messages from the agent


# ===============================
# 2. DSPy agent and streaming
# ===============================

agent_case_study = create_case_study_agent()
agent_chat = create_chat_agent()


class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    """
    Status message provider that emits structured, parseable messages
    so we can reconstruct tool calls cleanly.

    We use a simple "prefix::field::field" format.

    TOOL_START::<tool_name>::<inputs_repr>
    TOOL_END::<outputs_repr>
    """

    def tool_start_status_message(self, instance: Any, inputs: Dict[str, Any]) -> str:
        return f"TOOL_START::{instance.name}::{inputs}"

    def tool_end_status_message(self, outputs: Any) -> str:
        return f"TOOL_END::{outputs}"


agent_case_study = create_case_study_agent()
agent_chat = create_chat_agent()

# Separate listeners for each agent
stream_listeners_case = [
    dspy.streaming.StreamListener(
        signature_field_name="next_thought",
        # no allow_reuse needed here
    )
]

stream_listeners_chat = [
    dspy.streaming.StreamListener(
        signature_field_name="next_thought",
    )
]

stream_case_study = dspy.streamify(
    agent_case_study,
    status_message_provider=MyStatusMessageProvider(),
    stream_listeners=stream_listeners_case,
    async_streaming=False,
)

stream_chat = dspy.streamify(
    agent_chat,
    status_message_provider=MyStatusMessageProvider(),
    stream_listeners=stream_listeners_chat,
    async_streaming=False,
)


# ===============================
# 3. Trace render helpers
# ===============================


def render_trace_static(trace: TraceData) -> None:
    """
    For old assistant messages, re-render the stored trace in the same style:
      - expander
      - columns: thinking on left, per tool expanders on right
    """
    with st.expander("Agent thinking and tools", expanded=False):
        col_thinking, col_tools = st.columns(2)

        with col_thinking:
            thinking_text = trace["thinking"] or "(no thinking streamed)"
            st.markdown(f"**Thinking:** {thinking_text}")

        with col_tools:
            tools = trace["tools"]
            if not tools:
                st.markdown("_No tools were called._")
            else:
                for t in tools:
                    with st.expander(f"Tool: {t['name']}", expanded=False):
                        st.markdown("**Inputs:**")
                        st.code(t["inputs"], language="text")
                        st.markdown("**Output:**")
                        st.code(t["outputs"], language="text")


def collect_trace_from_stream(
    output_stream,
) -> Tuple[Optional[dspy.Prediction], TraceData]:
    """
    Pure helper that consumes a streamified agent output and builds a TraceData.
    No Streamlit in here.

    The format is driven by MyStatusMessageProvider above.
    """
    final_pred: Optional[dspy.Prediction] = None
    accumulated_thought = ""
    tools: List[ToolTrace] = []

    for chunk in output_stream:
        # Streaming tokens of the ReAct "next_thought" field
        if isinstance(chunk, dspy.streaming.StreamResponse):
            if chunk.signature_field_name == "next_thought":
                accumulated_thought += chunk.chunk

        # Status messages for tools
        elif isinstance(chunk, dspy.streaming.StatusMessage):
            msg = chunk.message or ""
            if msg.startswith("TOOL_START::"):
                _, name, inputs = msg.split("::", 2)
                tools.append(ToolTrace(name=name, inputs=inputs, outputs=""))
            elif msg.startswith("TOOL_END::"):
                _, outputs = msg.split("::", 1)
                if tools:
                    tools[-1]["outputs"] = outputs

        elif isinstance(chunk, dspy.Prediction):
            final_pred = chunk

    trace: TraceData = {"thinking": accumulated_thought, "tools": tools}
    return final_pred, trace


def stream_react_into_ui(
    query_value: str,
    stream_fn,
    extra_inputs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[dspy.Prediction], TraceData]:
    if extra_inputs is None:
        extra_inputs = {}
    with st.expander("Agent thinking and tools", expanded=True):
        col_thinking, col_tools = st.columns(2)
        with col_thinking:
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("**Thinking:** ")
        with col_tools:
            tools_container = st.container()
            tool_output_placeholders: List[st.delta_generator.DeltaGenerator] = []

    # call the chosen stream function
    output_stream = stream_fn(query=query_value, **extra_inputs)

    final_pred: Optional[dspy.Prediction] = None
    accumulated_thought = ""
    tools: List[ToolTrace] = []

    for chunk in output_stream:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            if chunk.signature_field_name == "next_thought":
                accumulated_thought += chunk.chunk
                thinking_placeholder.markdown(f"**Thinking:** {accumulated_thought}")
        elif isinstance(chunk, dspy.streaming.StatusMessage):
            msg = chunk.message or ""
            if msg.startswith("TOOL_START::"):
                _, name, inputs = msg.split("::", 2)
                tools.append(ToolTrace(name=name, inputs=inputs, outputs=""))
                with tools_container.expander(f"Tool: {name}", expanded=False):
                    st.markdown("**Inputs:**")
                    st.code(inputs, language="text")
                    st.markdown("**Output:**")
                    out_placeholder = st.empty()
                    out_placeholder.markdown("_Running..._")
                    tool_output_placeholders.append(out_placeholder)
            elif msg.startswith("TOOL_END::"):
                _, outputs = msg.split("::", 1)
                if tools:
                    tools[-1]["outputs"] = outputs
                if tool_output_placeholders:
                    tool_output_placeholders[-1].code(outputs, language="text")
        elif isinstance(chunk, dspy.Prediction):
            final_pred = chunk

    trace: TraceData = {"thinking": accumulated_thought, "tools": tools}
    return final_pred, trace


def build_history_string(messages: List[ChatMessage]) -> str:
    lines = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ===============================
# 4. Streamlit app: chat loop
# ===============================

st.set_page_config(page_title="Legal Contract and Agreements Agent", page_icon="LMA")

st.title("Legal Contract and Agreements Agent")
st.write(
    "Ask a question related to contracts or agreements. "
    "The agent will search your legal index, call tools, and build a case study."
)

# Initialize separate histories
if "messages_case_study" not in st.session_state:
    st.session_state.messages_case_study: List[ChatMessage] = []
if "messages_chat" not in st.session_state:
    st.session_state.messages_chat: List[ChatMessage] = []

# --- Agent selector + history + single bottom input ---

# Choose which agent / conversation is active
mode = st.radio(
    "Choose agent",
    ["Case study", "Conversational"],
    horizontal=True,
)

# Render history for the selected agent
if mode == "Case study":
    st.write(
        "Ask a question related to contracts or agreements. "
        "The agent will search your legal index, call tools, and build a case study."
    )
    for msg in st.session_state.messages_case_study:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("role") == "assistant" and "trace" in msg:
                render_trace_static(msg["trace"])  # type: ignore[arg-type]

elif mode == "Conversational":
    st.write("Conversational agent that sees previous messages as context.")
    for msg in st.session_state.messages_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("role") == "assistant" and "trace" in msg:
                render_trace_static(msg["trace"])  # type: ignore[arg-type]

# Single input, pinned at bottom (not inside tabs / columns / containers)
prompt = st.chat_input(
    "Ask something…"
    if mode == "Conversational"
    else "Ask something like: `Financial repercussions for delay in fulfilling Conditions Precedent`"
)

if prompt:
    if mode == "Case study":
        # ----- case-study flow (your old user_query branch) -----
        st.session_state.messages_case_study.append(
            ChatMessage(role="user", content=prompt)
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            pred, trace = stream_react_into_ui(
                query_value=prompt,
                stream_fn=stream_case_study,
            )
            if pred is not None and hasattr(pred, "case_study_md"):
                answer_text = str(pred.case_study_md)
            elif pred is not None and hasattr(pred, "answer"):
                answer_text = str(pred.answer)
            else:
                answer_text = "Agent did not return a case study."

            st.markdown(answer_text)

        st.session_state.messages_case_study.append(
            ChatMessage(role="assistant", content=answer_text, trace=trace)
        )

    else:
        # ----- conversational flow (your old chat_input branch) -----
        st.session_state.messages_chat.append(ChatMessage(role="user", content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        history_str = build_history_string(st.session_state.messages_chat[:-1])

        with st.chat_message("assistant"):
            pred, trace = stream_react_into_ui(
                query_value=prompt,
                stream_fn=stream_chat,
                extra_inputs={"history": history_str},
            )
            if pred is not None and hasattr(pred, "answer"):
                answer_text = str(pred.answer)
            else:
                answer_text = "Agent did not return an answer."

            st.markdown(answer_text)

        st.session_state.messages_chat.append(
            ChatMessage(role="assistant", content=answer_text, trace=trace)
        )
