from typing import Literal

from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.default.prompt_templates import HITL_TOOLS_PROMPT
from email_assistant.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_hitl, default_background, default_triage_instructions, default_response_preferences, default_cal_preferences
from email_assistant.schemas import State, RouterSchema, StateInput
from email_assistant.utils import parse_email, format_for_display, format_email_markdown
from dotenv import load_dotenv

load_dotenv(".env")

# Get tools
tools = get_tools(["write_email", "schedule_meeting", "check_calendar_availability", "Question", "Done"])
tools_by_name = get_tools_by_name(tools)

# Initialize the LLM for use with router / structured output
llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_router = llm.with_structured_output(RouterSchema) 

# Initialize the LLM, enforcing tool use (of any available tools) for agent
llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_with_tools = llm.bind_tools(tools, tool_choice="required")


# Nodes
# create the same graph but now with interrupts
def triage_router(state: State) -> Command[Literal["__end__", "triage_interrupt_handler", "response_agent"]]:
    """ Analyze the email content to decide if we should respond, notify, or ignore the email """

    # gets the email as input and has to do some classification on it to decide to 

    # Parse the email input 
    author, to, subject, email_thread = parse_email(state["email_input"])

    user_prompt = triage_user_prompt.format(author=author, to=to, subject=subject, email_thread=email_thread)

    email_markdown = format_email_markdown(subject, author, to, email_thread)

    system_prompt = triage_system_prompt.format(background=default_background, triage_instructions=default_triage_instructions)

    # Run the router llm to ge the classification based on the system and user prompt added for context to make the correct
    # decision

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Decision
    classification = result.classification # the result.classification is because of the pydantic output structure

    # Process each classification result
    if classification == "respond":
        # we want to update the classification state and send to the response agent node
        goto = "response_agent"

        update = {
            "classification_decision": classification,
            # letting the agent know to respond to the message
            "messages": [
                {"role": "user", "content": f"Respond to the email: {email_markdown}"}
            ]

        }
    elif classification == "ignore":
        goto = END
        # Update the state
        update = {
            "classification_decision": classification
        }
    elif classification == "notify":
        # we want to go to the interrupt node where it can ask the user to decide to handle the email or not.

        # what context do we want to update the state with ?
        # update the classification result and now do we add a messages "email under review for the user"

        goto = "triage_interrupt_handler"

        # update the state
        update = {
            "classification_decision": classification
        }
    else:
        raise ValueError(f"Invalid classification: {classification}")
    return Command(goto=goto, update=update)
    

def triage_interrupt_handler(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Interrupt handler: ask the user whether to proceed or ignore."""
    author, to, subject, email_thread = parse_email(state["email_input"])
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    request = {
        "action_request": {"action": f"Email Assistant: {state['classification_decision']}", "args": {}},
        "config": {"allow_ignore": True, "allow_respond": True, "allow_edit": False, "allow_accept": False},
        "description": email_markdown,
    }

    _resp = interrupt([request])
    response = _resp[0] if isinstance(_resp, list) else _resp
    if isinstance(response, str):
        response = {"type": response}

    if response.get("type") == "response":
        user_input = response.get("args")
        goto = "response_agent"
        update = {
            "messages": [
                {"role": "user", "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}"}
            ]
        }
    elif response.get("type") == "ignore":
        goto = END
        update = {}
    else:
        raise ValueError(f"Invalid response: {response}")

    return Command(goto=goto, update=update)


def llm_call(state: State):
    """LLM decides whether to call a tool or not."""
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {
                        "role": "system",
                        "content": agent_system_prompt_hitl.format(
                            tools_prompt=HITL_TOOLS_PROMPT,
                            background=default_background,
                            response_preferences=default_response_preferences,
                            cal_preferences=default_cal_preferences,
                        ),
                    }
                ]
                + state["messages"]
            )
        ]
    }

def interrupt_handler(state: State) -> Command[Literal["__end__", "llm_call"]]:
    """ Creates an interrupt for human review of tool calls """

    # store the result of the tool calls
    result = []

    # we need the last message contain the action
    last_message = state["messages"][-1]

    # Go to the LLM call node next

    goto = "llm_call"

    for tool_call in last_message.tool_calls:

        # Allowed tools for HITL
        hitl_tools = ["write_email", "schedule_meeting", "Question"]

        if tool_call["name"] not in hitl_tools:
            # we execute as normal

            action_args = tool_call["args"]

            # new observation
            observation = tools_by_name[tool_call["name"]].invoke(action_args)

            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
            
            # continue to the next tool call
            continue
        

        # now this for all the tools that trigger the HITL interrupt

        # Get original email from email_input in state
        email_input = state["email_input"]
        author, to, subject, email_thread = parse_email(email_input)
        original_email_markdown = format_email_markdown(subject, author, to, email_thread)
        
        # Format tool call for display and prepend the original email
        tool_display = format_for_display(tool_call)
        description = original_email_markdown + tool_display


        # now set up the config for each tool call
        # Configure what actions are allowed in Agent Inbox

        if tool_call["name"] == "write_email":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True
            }
        elif tool_call["name"] == "schedule_meeting":

            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True
            }
        elif tool_call["name"] == "Question":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False
            }
        else:
            raise ValueError(f"Invalid tool call: {tool_call['name']}")
        

        # Create the interrupt request
        request = {
            "action_request": {
                "action": tool_call["name"],
                "args": tool_call["args"]
            },
            "config": config,
            "description": description,
        }

        # Send to Agent Inbox and wait for response
        _resp = interrupt([request])
        response = _resp[0] if isinstance(_resp, list) else _resp
        if isinstance(response, str):
            response = {"type": response}

        if response.get("type") == "response":
            # we want to add the response to the messages

            # user provided feedback
            user_feedback = response.get("args")

            if tool_call["name"] == "write_email":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            elif tool_call["name"] == "schedule_meeting":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            elif tool_call["name"] == "Question":
                # Don't execute the tool, and add a message with the user feedback to incorporate into the email
                result.append({"role": "tool", "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")
        elif response.get("type") == "ignore":
            if tool_call["name"] == "write_email":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool", "content": "User ignored this email draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
            elif tool_call["name"] == "schedule_meeting":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool", "content": "User ignored this calendar meeting draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
            elif tool_call["name"] == "Question":
                # Don't execute the tool, and tell the agent how to proceed
                result.append({"role": "tool", "content": "User ignored this question. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                # Go to END
                goto = END
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")
        elif response.get("type") == "edit":
            # we want to update the tool call with the updates args

            # Tool selection
            tool = tools_by_name[tool_call["name"]]

            # Get the edited args from the user
            args_obj = response.get("args")
            edited_args = args_obj["args"] if isinstance(args_obj, dict) and "args" in args_obj else args_obj

            # Update the AI message's tool call with edited content (reference to the message in the state)
            ai_message = state["messages"][-1] # Get the most recent message from the state
            current_id = tool_call["id"] # Store the ID of the tool call being edited
            
            # Create a new list of tool calls by filtering out the one being edited and adding the updated version
            # This avoids modifying the original list directly (immutable approach)
            updated_tool_calls = [tc for tc in ai_message.tool_calls if tc["id"] != current_id] + [
                {"type": "tool_call", "name": tool_call["name"], "args": edited_args, "id": current_id}
            ]

            # Create a new copy of the message with updated tool calls rather than modifying the original
            # This ensures state immutability and prevents side effects in other parts of the code
            result.append(ai_message.model_copy(update={"tool_calls": updated_tool_calls}))


            # Update the write_email tool call with the edited content from Agent Inbox
            if tool_call["name"] == "write_email":

                # Execute the tool with edited args
                observation = tool.invoke(edited_args)
                
                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

            # Update the schedule_meeting tool call with the edited content from Agent Inbox
            elif tool_call["name"] == "schedule_meeting":
                
                
                # Execute the tool with edited args
                observation = tool.invoke(edited_args)
                
                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})
            
            # Catch all other tool calls
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response.get("type") == "accept":
            # we want to execute the tool

            # Tool selection / Action that is chosen
            tool = tools_by_name[tool_call["name"]]

            # Execute the tool with the args
            observation = tool.invoke(tool_call["args"])

            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})

        else:
            raise ValueError(f"Invalid response: {response}")
        
    # Update the state 
    update = {
        "messages": result,
    }

    return Command(goto=goto, update=update)


# Conditional edge function
def should_continue(state: State) -> Literal["interrupt_handler", "__end__"]:
    """Route to tool handler, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls: 
            if tool_call["name"] == "Done":
                return END
            else:
                return "interrupt_handler"

# Build workflow
agent_builder = StateGraph(State)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        END: END,
    },
)

# Compile the agent
response_agent = agent_builder.compile()


# Build overall workflow
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
    
)

email_assistant = overall_workflow.compile()
        