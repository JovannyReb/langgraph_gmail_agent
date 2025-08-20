from typing import Literal

from langchain.chat_models import init_chat_model

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.default.prompt_templates import AGENT_TOOLS_PROMPT
from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt, default_background, default_triage_instructions, default_response_preferences, default_cal_preferences
from email_assistant.schemas import State, RouterSchema, StateInput
from email_assistant.utils import parse_email, format_email_markdown

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from dotenv import load_dotenv
import json
load_dotenv(".env")

# Get tools
tools = get_tools()

tools_by_name = get_tools_by_name(tools)


def _tool_to_dict(tool):
    schema = None
    try:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            # pydantic v2 preferred
            try:
                schema = args_schema.model_json_schema()
            except Exception:
                try:
                    # fallback for older pydantic versions
                    schema = args_schema.schema()
                except Exception:
                    schema = None
    except Exception:
        schema = None
    return {
        "name": getattr(tool, "name", None),
        "description": getattr(tool, "description", None),
        "args_schema": schema,
    }


_pretty = {name: _tool_to_dict(t) for name, t in tools_by_name.items()}
print(json.dumps(_pretty, indent=2, ensure_ascii=False, sort_keys=True))

# Initialize the LLM for use with router / structured output
llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_router = llm.with_structured_output(RouterSchema) 

# Initialize the LLM, enforcing tool use (of any available tools) for agent
llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_with_tools = llm.bind_tools(tools, tool_choice="any")


# first thing create the state (imported from email_assistant.schemas as State)

# create the riage_router
def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore."""

    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )

    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Run the router LLM
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification

    if result.classification == "respond":
        goto = "response_agent"
        
        # we then update the state with the new classification and add the email markdown to the messages
        update = {
            "classification_decision": classification,
            "messages": [
                {"role": "user", "content": email_markdown}
            ]
        }
        return Command(goto=goto, update=update)
    
    elif result.classification == "ignore":
        goto = END
        update = {"classification_decision": classification}
    elif result.classification == "notify":
        goto = END
        update = {"classification_decision": classification}
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)


# llm decides to call a tool or not
def llm_call(state: State):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            # Invoke the LLM that has the tools
            llm_with_tools.invoke(
                # Add the system prompt
                [
                    {
                        "role": "system",
                        "content": agent_system_prompt.format(
                            tools_prompt=AGENT_TOOLS_PROMPT,
                            background=default_background,
                            response_preferences=default_response_preferences,
                            cal_preferences=default_cal_preferences,
                        ),
                    }
                ]
                # Add the current messages to the prompt
                + state["messages"]
            )
        ]
    }


# Nodes
def tool_node(state: State):
    """Perform the tool call."""

    result = []

    # Iterate through tool calls
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool
        tool = tools_by_name[tool_call["name"]]
        # Run it
        observation = tool.invoke(tool_call["args"])
        # Create a tool message
        result.append(
            {"role": "tool", "content": observation, "tool_call_id": tool_call["id"]}
        )

    # Add it to our messages
    return {"messages": result}


# Conditional edge function
def should_continue(state: State) -> Literal["Action", "__end__"]:
    """Route to Action, or end if Done tool called."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                return END
            else:
                return "Action"


# build agent
agent = StateGraph(State)

# add nodes
agent.add_node("llm_call", llm_call)
agent.add_node("environment", tool_node)

agent.add_edge(START, "llm_call")
agent.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        END: END,
    },
)

agent.add_edge("environment", "llm_call")

agent.add_edge("llm_call", END)

# build the graph
agent_workflow = agent.compile()


# Build workflow, this is because triage_router is a function that returns a command
overall_workflow = (
    StateGraph(State, input_schema=StateInput)
    .add_node("triage_router", triage_router)
    .add_node("response_agent", agent_workflow)
    .add_edge(START, "triage_router")
)

# compile the graph
email_assistant = overall_workflow.compile()







    
