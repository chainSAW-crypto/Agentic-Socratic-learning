from Dependancies import init_graph_state, graph_runnable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(
    title="Conversational Graph API",
    description="API for processing messages through a conversational graph workflow",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "conversation",
            "description": "Operations related to conversation processing"
        },
        {
            "name": "system",
            "description": "System management operations"
        }
    ]
)


# Input payload model
class InputPayload(BaseModel):
    user_input: str

    class Config:
        schema_extra = {
            "example": {
                "user_input": "Hello, how can you help me today?"
            }
        }


# Output payload model
class OutputPayload(BaseModel):
    agent_response: str
    reference_documents: Optional[str] = None
    current_agent: str

    class Config:
        schema_extra = {
            "example": {
                "agent_response": "I'm here to assist you with your questions!",
                "reference_documents": "knowledge_base/general_info.txt",
                "current_agent": "assistant"
            }
        }


# Simple response for reset
class ResetResponse(BaseModel):
    status: str
    message: str

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Graph state has been reset to initial values"
            }
        }


# Initialize global state
state = init_graph_state()  # Assuming this is predefined


@app.post(
    "/process_message",
    response_model=OutputPayload,
    tags=["conversation"],
    summary="Process a user message",
    description="Takes a user message, processes it through the graph workflow, and returns the agent's response along with any retrieved documents"
)
async def process_message(payload: InputPayload):
    """
    Process a user message through the conversational graph workflow.

    - **user_input**: The text message from the user

    Returns an object containing:
    - **agent_response**: The text response from the agent
    - **reference_documents**: Any reference documents retrieved during processing
    - **current_agent**: The identifier of the current active agent
    """
    global state

    # Update state with new message
    state["messages"].append(HumanMessage(content=payload.user_input))

    # Run the graph
    new_state = graph_runnable.invoke(state)

    # Prepare output payload
    output = OutputPayload(
        agent_response=new_state["messages"][-1].content,
        reference_documents=new_state["retrieved_docs"][-1] if new_state["retrieved_docs"] else None,
        current_agent=new_state["current_agent"]
    )

    # Update the global state
    state = new_state

    return output


@app.post(
    "/reset_state",
    response_model=ResetResponse,
    tags=["system"],
    summary="Reset conversation state",
    description="Resets the graph state to its initial values, clearing all conversation history"
)
async def reset_state():
    """
    Reset the graph state to its initial values.

    This endpoint clears all conversation history, messages, and retrieved documents,
    effectively starting a new conversation.

    Returns a status message confirming the reset.
    """
    global state

    # Reset the state to initial values
    state = init_graph_state()

    return ResetResponse(
        status="success",
        message="Graph state has been reset to initial values"
    )


@app.get(
    "/",
    tags=["system"],
    summary="API Status",
    description="Checks if the API is running correctly"
)
async def root():
    """
    Root endpoint to check if the API is running.

    Returns a simple status message.
    """
    return {"status": "online", "message": "Conversational Graph API is running"}
