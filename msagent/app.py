__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
import chromadb
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")

# Search tool
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()

search_tool = Tool(
    name='google_search',
    description='Search google for recent results',
    func=search.run,
)

# Create ChromaDB client for PDF data
pdf_client = chromadb.PersistentClient(path="./chroma_db/pdf_data")

# Create the collection
try:
    collection = pdf_client.get_or_create_collection("pdf_docs")
    print(f"Collection 'pdf_docs' ready")
except Exception as e:
    print(f"Error creating collection: {e}")

# Create Chroma vector store for PDF documents
pdf_db = Chroma(
    client=pdf_client,
    collection_name="pdf_docs",
    embedding_function=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004'),
)

docs_retreiver = pdf_db.as_retriever()

pdfs = create_retriever_tool(
    retriever=docs_retreiver,
    name='pdf retreiver',
    description='Get educational content about open source applications'
)

# Create ChromaDB client for web data
web_client = chromadb.PersistentClient(path="./chroma_db/web_data")

# Create the collection
try:
    web_collection = web_client.get_or_create_collection("web_docs")
    print(f"Collection 'web_docs' ready")
except Exception as e:
    print(f"Error creating collection: {e}")

# Create Chroma vector store for web documents
web_db = Chroma(
    client=web_client,
    collection_name="web_docs",
    embedding_function=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004'),
)

web_retriever = web_db.as_retriever()

web_tool = create_retriever_tool(
    retriever=web_retriever,
    name='web_retriever',
    description='Get Unit purpose and description'
)

tools = [pdfs,
         search_tool,
         web_tool]

# Define a more structured prompt template
prompt = PromptTemplate.from_template("""You are University of Embu's expert educational assistant for open source applications, focused on helping students learn effectively.
You prioritize thorough understanding and clear explanations based on reliable course materials.

You have access to the following tools:
{tools}

STRATEGY GUIDELINES:
1. ALWAYS check course PDF materials FIRST - these contain the most relevant and authoritative information
2. Only use web search or other tools when the PDFs don't contain sufficient information
3. When explaining concepts, include relevant examples and relate to real-world applications
4. Break down complex topics into manageable parts
5. If multiple sources provide different perspectives, synthesize them and explain the variations

You must follow this exact format:

Question: the input question you must answer
Thought: your reasoning about what to do next (be thorough in your thinking)
Action: the tool name to use (must be one of: {tool_names})
Action Input: the input to pass to the tool
Observation: the result from the tool
... (you can repeat the Thought/Action/Action Input/Observation steps multiple times)
Thought: your final reasoning - synthesize what you've learned and organize your response
Final Answer: your comprehensive educational response that includes:
  - Clear explanation of concepts
  - Examples when helpful
  - Citations to course materials when applicable
  - Summary of key points

Begin! Remember to ALWAYS follow the format exactly and prioritize course PDF materials before using other tools.

Question: {input}
{agent_scratchpad}
""")


memory = ChatMessageHistory(session_id="test-session")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Create the agent - will choose a sequence of actions to take using the tools based on the query
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Executor to execute the agent

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

agent_with_chat_history = RunnableWithMessageHistory(
    executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)


st.title('Study Buddy - Open Source Apps UoEm')
st.markdown("By: *Rickmwasofficial* - **For Educational Purposes only**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input('Where is this unit taught?'):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({'role': "user", "content": prompt})

    response = agent_with_chat_history.invoke({'input': prompt},
                               config={"configurable": {"session_id": "<foo>"}},)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if response.get("intermediate_steps"):
            for step in response["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) > 0:
                    agent_action = step[0]
                    if hasattr(agent_action, "log"):
                        # Extract just the "Thought:" part
                        thought_log = agent_action.log
                        if "Thought:" in thought_log:
                            thought_text = thought_log.split("Action:")[0].strip()
                            st.markdown(thought_text)
            st.markdown("### Response: ")
        
        st.markdown(response['output'])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['output']})
