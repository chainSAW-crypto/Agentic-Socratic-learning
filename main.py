from Dependancies import init_graph_state, graph_runnable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Input payload model
class InputPayload(BaseModel):
    user_input: str

# Output payload model
class OutputPayload(BaseModel):
    agent_response: str
    reference_documents: Optional[str] = None
    current_agent: str

# Global state management (you'll replace this with your actual state management)
state = init_graph_state()  # Assuming this is predefined

@app.post("/process_message", response_model=OutputPayload)
async def process_message(payload: InputPayload):
    global state  # Declare global here, before using it

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



class ResetResponse(BaseModel):
    status: str
    message: str

@app.post("/reset_state", response_model=ResetResponse)
async def reset_state():
    global state

    # Reset the state to initial values
    state = init_graph_state()

    return ResetResponse(
        status="success",
        message="Graph state has been reset to initial values"
    )
