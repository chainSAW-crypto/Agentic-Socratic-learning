# !pip install langgraph langsmith langchain langchain_groq langchain_community python-decouple langchain_huggingface

import os
from langchain_community.tools import TavilySearchResults
from typing import Dict, List, TypedDict, Annotated, Sequence, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
import operator
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import re
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import Literal
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Sequence
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq


load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
groq_api_key = os.getenv("groq_api_key")

os.environ['HF_TOKEN']= os.getenv("HF_TOKEN")
embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# state Class Function
class GraphState(TypedDict):  # Class GraphState
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    conversation_history: Dict[str, List[BaseMessage]]
    retrieved_docs: List[str]
    current_llm_model: str
    context_summary: str
    pdf_path: str



def process_pdf(pdf_path):
    """
    Processes a PDF document by loading, splitting, embedding, and storing it in a vector database.
    Returns a retriever for querying relevant document chunks.

    Parameters:
    - pdf_path (str): Path to the PDF document.
    - embedding_model (str): The embedding model to use (default: "text-embedding-ada-002").

    Returns:
    - retriever: A retriever object for fetching relevant chunks from the document.
    """

    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Store in ChromaDB and create a retriever
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever


# Tavily agent Function
search_tool = TavilySearchResults(
    max_results=5
)
def agent_tavily(state):
  messages = search_tool.invoke(state["question"])
  state["messages"] = [f"source url: {c['url']} \n content: {c['content']}" for c in messages]

  return {"messages": state["messages"]}




# Function to get current user query from the Graph State

def get_current_user_question(state: GraphState) -> str:
    """Extract the most recent user question from the state."""
    # Find the last HumanMessage in the messages list
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content

    # If no HumanMessage is found, return empty string
    return ""

def get_conversation_history(state: GraphState) -> Dict[int, Dict[str, Any]]:
    """Retrieves the complete conversation history from the state."""
    return state["messages"]

# Example usage:
# conversation_history = get_conversation_history(state)
# print(conversation_history)


def summarize_conversation(state):
    """
    Summarizes the entire conversation history into a concise context summary
    for the Socratic LLM to work with, keeping key details while reducing token usage.

    Parameters:
    - state: The conversation state containing history.
    - groq_api_key: API key for the Groq model.
    - model: The LLM model to use for summarization (default: "llama-3.1-8b-instant").

    Returns:
    - A summarized context of the conversation.
    """

    context_summarizer_model = "llama-3.1-8b-instant"
    context_summarizer_llm = ChatGroq(groq_api_key=groq_api_key, model=context_summarizer_model)

    # System prompt for summarization
    context_summarizer_system = """
    You are a context summarization agent responsible for condensing a given conversation history
    while retaining all important details. Your goal is to create a concise yet comprehensive
    summary that preserves the critical context for the Socratic LLM to work effectively.

    Guidelines:
    - Prioritize Key Information: Extract the most relevant details, such as questions asked,
      user challenges, partial answers provided, and the direction of the conversation.
    - Maintain Logical Flow: Ensure the summary follows a structured progression, making it easy
      for the Socratic LLM to understand the conversation history.
    - Preserve Intent and Meaning: Do not omit crucial information that might change the meaning
      of the discussion. Avoid unnecessary repetition.
    - Limit Length: Keep the summary under 2500 tokens while maintaining readability and coherence.
    - Condense Without Losing Clarity: Use concise wording, bullet points, and well-structured
      sentences to optimize space.
    - Adapt Based on Content Type: If the conversation involves coding, math, or technical topics,
      include key concepts, partial solutions, and problem-solving approaches discussed.

    Example Output:
    Input Conversation History:
    User: "How do I reverse a string in Python?"
    Agent: "Have you considered using slicing?"
    User: "I don’t understand."
    Agent: "Think about how indexing works. How would you select the last character?"
    User: "Negative indexing?"
    Agent: "Exactly! Now, how can you use that for the whole string?"
    User: "Still confused."
    Agent: "Try using slicing like [::-1]."

    Summarized Output:
    User wants to reverse a string in Python. The agent guided them toward slicing but they
    struggled with indexing. The agent introduced negative indexing as a hint, leading to
    the final solution ([::-1]).
    """

    # Get conversation history
    full_conv_history = get_conversation_history(state)
    formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in full_conv_history])

    # Create the summarization prompt
    context_summarizer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_summarizer_system),
            ("human", f"Please create a concise context summary of the following conversation information:\n\n{formatted_history}"),
        ]
    )

    # Generate the summarized context
    #summarized_context = (context_summarizer_prompt | context_summarizer_llm).invoke(full_conv_history)
    summarized_context = (context_summarizer_prompt | context_summarizer_llm).invoke({"full_conv_history": formatted_history})  # Pass as a dictionary

    return summarized_context



socratic_llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")
system = """You are a Socratic AI Tutor, an expert in all fields, including mathematics, coding, science, philosophy, and more. Your goal is not to provide direct answers but to guide the user toward discovering the answer themselves.
You will have access to all the conversation history so please take in account the context also.

Guidelines:
Encourage Thinking: When the user asks a question, respond with thought-provoking hints, leading questions, or analogies instead of a direct answer.
Adapt to Difficulty Level: If the user struggles or asks for more help, gradually provide clearer hints.
Final Answer as a Last Resort: Only if the user is completely stuck and explicitly asks for the answer, provide it concisely.
Engage in Dialogue: Keep responses engaging, Socratic-style, ensuring the user actively thinks rather than passively receives information.
Support All Fields: Be a master of every domain, adapting hints and reasoning styles accordingly—step-by-step logic for coding/math, conceptual guidance for philosophy, etc.
Encourage Learning Mindset: Praise effort, reinforce curiosity, and help the user build problem-solving skills.
"""

socratic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

socratic_agent = socratic_prompt | socratic_llm

def socratic_agent_handler(state: GraphState):

    question = state["messages"][-1].content
    context = summarize_conversation(state)

    inbuilt_query = f""" You are a Socratic AI Tutor, an expert in all fields, including mathematics, coding, science, philosophy, and more. Your goal is not to provide direct answers but to guide the user toward discovering the answer themselves.
    You will have access to all the conversation history so please take in account the prvided context also.
    conversation chat context : {context}
    user query: {question}
    """
    response = socratic_agent.invoke(inbuilt_query)

    # Add the question-response pair to conversation_history
    conversation_id = len(state["conversation_history"]) + 1
    state["conversation_history"][conversation_id] = {
        "question": question,
        "response": response
    }

    state["llm_model"] = "llama-3.3-70b-versatile"

    # Also add the AI response to the messages list
    state["messages"].append(AIMessage(content=response.content))

    # Update the current_agent within the state dictionary
    # state['current_agent'] = "moderator_agent"  # Or the appropriate agent name

    # Return the updated state
    return state


# Coding Agent handler
socratic_coding_llm = ChatGroq(groq_api_key=groq_api_key, model="qwen-2.5-coder-32b")
coding_system_prompt = """You are a Socratic Coding Tutor, an expert in programming and computer science concepts. Your role is to guide students through coding problems using the Socratic method, helping them develop problem-solving skills and deep understanding rather than providing direct answers.

Specialized Guidelines for Coding:
1. Problem Decomposition: Help students break down coding problems into smaller, manageable parts
2. Debugging Guidance: When errors occur, ask questions that lead students to discover the issue themselves
3. Conceptual Understanding: Focus on underlying concepts rather than syntax alone
4. Algorithmic Thinking: Guide students to think about efficiency and different approaches
5. Step-by-Step Discovery: Use leading questions to help students arrive at solutions incrementally

Teaching Approach:
- Start with high-level questions about the problem statement
- Progress to more specific questions about implementation
- For stuck students, suggest exploring specific concepts or techniques
- Use analogies from real-world systems when explaining abstract concepts
- Encourage testing and experimentation with small code examples
- Only provide direct code solutions as a last resort

Domains of Expertise:
- Programming languages (Python, Java, JavaScript, etc.)
- Data structures and algorithms
- Software design patterns
- Debugging techniques
- Computational thinking
"""

coding_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", coding_system_prompt),
        ("human", "{question}"),
    ]
)

socratic_coding_agent = coding_prompt | socratic_coding_llm


def coding_agent_handler(state: GraphState):
    question = state["messages"][-1].content
    context = summarize_conversation(state)

    coding_query = f"""As a Socratic Coding Tutor, use the conversation context to guide the student toward solving their coding problem through thoughtful questioning.

    Conversation Context: {context}

    Current Problem: {question}

    Consider these steps in your response:
    1. Identify the core programming concept involved
    2. Determine where the student might be stuck
    3. Formulate questions that:
       - Clarify any misunderstandings
       - Suggest relevant programming concepts to consider
       - Guide toward breaking down the problem
       - Encourage testing small parts of the solution
    4. If appropriate, suggest pseudocode or high-level steps without giving complete solutions
    """

    response = socratic_coding_agent.invoke(coding_query)

    # Add the question-response pair to conversation_history
    conversation_id = len(state["conversation_history"]) + 1
    state["conversation_history"][conversation_id] = {
        "question": question,
        "response": response
    }

    state["llm_model"] = "qwen-2.5-coder-32b"  # Keep the coding-specific model

    # Also add the AI response to the messages list
    state["messages"].append(AIMessage(content=response.content))

    return state


class routetollms(BaseModel):
    datasource: Literal["socratic_agent", "socratic_coding_agent"] = Field(
        ...,
        description="""Based on the user question and the nature of task route the query to socratic coding agent or socratic agent. socratic coding agent is specilized on coding and programming related problems, 
      while socratic agent specilized in more generalized information and other all types of problems.""",
    )


llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
llm_router = llm.with_structured_output(routetollms)

system = """You are an expert at routing a user question to respective LLM agent based on the provided query.
   The agents are socratic coding agent or socratic agent based on the given question and scenario provided by the user.
   Guide the user to the right agent.
   Your response must be one of the following:
   - socratic_agent
   - socratic_coding_agent
   """

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | llm_router


def route_questions(state: GraphState):
    """Route the query to the appropriate agent based on content analysis."""

    print("---ROUTE QUESTION---")
    question = get_current_user_question(state)  # Getting the question from the User Datastore
    source = question_router.invoke({"question": question})

    if source.datasource == "socratic_agent":
        print("---ROUTE QUESTION TO Socratic Agent---")
        return "socratic_agent"
    elif source.datasource == "socratic_coding_agent":
        print("---ROUTE QUESTION TO Socratic Coding Agent---")
        return "socratic_coding_agent"



from langgraph.graph import StateGraph, END, START


# Initialize the graph
workflow = StateGraph(GraphState)
# Add the nodes
#workflow.add_node("socratic_doc_agent", socratic_doc_agent_handler)
workflow.add_node("socratic_agent", socratic_agent_handler)
workflow.add_node("socratic_coding_agent", socratic_agent_handler)

# Add conditional edges - Fix: use START instead of "start"
workflow.add_conditional_edges(
    START,  # Changed from "start" to START
    route_questions,
    {
        #"socratic_doc_agent": "socratic_doc_agent",
        "socratic_agent": "socratic_agent",
        "socratic_coding_agent": "socratic_coding_agent"
    }
)

# Add edges from agents back to END
# workflow.add_edge("socratic_doc_agent", END)

workflow.add_edge("socratic_agent", END)
workflow.add_edge("socratic_coding_agent", END)

# Compile the graph

graph_runnable = workflow.compile()


# Initialize state
def init_graph_state() -> GraphState:
    return GraphState(
        messages=[],
        current_agent="",
        conversation_history={},
        retrieved_docs=[],
        current_llm_model="",
        context_summary=" "
    )