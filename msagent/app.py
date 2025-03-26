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
from langchain.memory import ChatMessageHistory
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
import chromadb
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

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
cnet_client = chromadb.PersistentClient(path="./chroma_db/pdf_data/cnet")
dbms_client = chromadb.PersistentClient(path="./chroma_db/pdf_data/dbms")
edp_client = chromadb.PersistentClient(path="./chroma_db/pdf_data/edp")
mis_client = chromadb.PersistentClient(path="./chroma_db/pdf_data/mis")
open_source_client = chromadb.PersistentClient(path="./chroma_db/pdf_data/open_source")
research_client = chromadb.PersistentClient(path="./chroma_db/pdf_data/research")
swe_client = chromadb.PersistentClient(path="./chroma_db/pdf_data/swe")

def create_pdf_retreival(client, name):
    # Create the collection
    try:
        collection = client.get_or_create_collection("pdf_docs")
        print(f"Collection 'pdf_docs' ready")
    except Exception as e:
        print(f"Error creating collection: {e}")

    # Create Chroma vector store for PDF documents
    pdf_db = Chroma(
        client=client,
        collection_name="pdf_docs",
        embedding_function=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004'),
    )

    docs_retreiver = pdf_db.as_retriever()

    pdfs = create_retriever_tool(
        retriever=docs_retreiver,
        name=name,
        description=f'Get educational content about {name}'
    )

    return pdfs

cnet = create_pdf_retreival(cnet_client, "computer_networks")
dbms = create_pdf_retreival(dbms_client, "database_management")
edp = create_pdf_retreival(edp_client, "event_driven")
mis = create_pdf_retreival(mis_client, "information_systems_management")
open_source = create_pdf_retreival(open_source_client, "open_source")
research = create_pdf_retreival(research_client, "research_methods")
swe = create_pdf_retreival(swe_client, "software_engineering")

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

# Define the retriever tool
def search_embuni(query):
    """Retrieve relevant content from the Embuni e-learning platform."""
    docs = web_retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

web_tool = Tool(
    func=search_embuni,
    name='web_retriever',
    description='Get Unit purpose and description, lecturer name etc..'
)

tools = [cnet,
         dbms,
         edp,
         mis,
         open_source,
         research,
         swe,
         search_tool,
         web_tool]

# Define a more structured prompt template
prompt = PromptTemplate.from_template("""You are University of Embu's expert educational assistant for second year units, focused on helping students learn effectively.
You prioritize thorough understanding and clear explanations based on reliable course materials.

You have access to the chat history to provide more relevant answers:
{history}
                                      
The units are:
 - Open Source Applications, Computer Networks, Database management systems, event driven programming, information system management, open source applications, research methods and software engineering.
You have access to the following tools:
{tools}

STRATEGY GUIDELINES:
1. ALWAYS check course PDF materials FIRST - these contain the most relevant and authoritative information
2. Only use web search or other tools when the PDFs don't contain sufficient information
3. When explaining concepts, include relevant examples and relate to real-world applications
4. Break down complex topics into manageable parts
5. If multiple sources provide different perspectives, synthesize them and explain the variations
6. You can use web search to add more information to the content available in the documents
7. The second priority after pdfs is checking from the web based agent tool

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

st.title('Study Buddy - Second Year Units')
st.markdown("By: *Rickmwasofficial* - **For Educational Purposes only**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat memory
if "memory" not in st.session_state:
    # Create a ChatMessageHistory object first
    message_history = ChatMessageHistory()
    
    # Then use it in the ConversationBufferMemory
    st.session_state.memory = ConversationBufferMemory(
        chat_memory=message_history,
        memory_key="chat_history",
        return_messages=True
    )

agent_with_chat_history = RunnableWithMessageHistory(
    executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: st.session_state.memory.chat_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

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

    response = agent_with_chat_history.invoke(
                {"input": prompt, "history": st.session_state.memory}, 
                {"configurable": {"session_id": "default"}}
            )
    print(st.session_state.memory)

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
