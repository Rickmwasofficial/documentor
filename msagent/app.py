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

from streamlit_chat import message


st.set_page_config(page_title="Blue - Studdy Buddy")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

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

# YouTube Search Tool
from googleapiclient.discovery import build

def search_youtube_videos(query):
    """Search for relevant YouTube videos."""
    youtube = build('youtube', 'v3', developerKey=os.getenv("GOOGLE_API_KEY"))
    
    request = youtube.search().list(
        part="snippet",
        maxResults=3,  # Explicitly limit to top 3 videos
        q=query + " tutorial",  # Add tutorial to improve educational content
        type="video",
        videoEmbeddable="true",  # Ensure videos can be embedded    
    )
    response = request.execute()
    
    # Extract video details
    videos = []
    for item in response['items']:
        videos.append({
            'title': item['snippet']['title'],
            'video_id': item['id']['videoId'],
            'embed_link': f"https://www.youtube.com/embed/{item['id']['videoId']}"
        })
    
    return videos

youtube_tool = Tool(
    func=search_youtube_videos,
    name='youtube_search',
    description='Search for relevant educational YouTube videos'
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
         web_tool,
         youtube_tool]

# Define a more structured prompt template
prompt = PromptTemplate.from_template("""You are Blue, University of Embu's expert educational assistant for second year units, focused on helping students learn effectively.
You prioritize thorough understanding and clear explanations based on reliable course materials.

You have access to the chat history to provide more relevant answers:
{history}
                                      
The units are:
 - Open Source Applications, Computer Networks, Database management systems, event driven programming, information system management, open source applications, research methods and software engineering.
You have access to the following tools:
Only use the youtube tools for relevant situation like providing further info to the students but not in all replies
{tools}

STRATEGY GUIDELINES:
1. Try and establish a rapport with the user by responding to greetings, asking for the name, and referring to the user by name when in a conversation
2. ALWAYS check course PDF materials FIRST - these contain the most relevant and authoritative information
3. Only use web search or other tools when the PDFs don't contain sufficient information
4. When explaining concepts, include relevant examples and relate to real-world applications
5. Break down complex topics into manageable parts
6. If multiple sources provide different perspectives, synthesize them and explain the variations
7. You can use web search to add more information to the content available in the documents
8. CRITICAL YOUTUBE RECOMMENDATION RULE:
    ONLY use the youtube_tool WHEN:
    - Query involves a clear educational concept
    - Seeks detailed explanation of a technical topic
    - Requires in-depth understanding of a specific subject
    DO NOT suggest videos for:
    - Greetings (hi, hello, how are you)
    - Personal introductions
    - Simple conversational exchanges
    - One-word or very short queries
    - Requests that don't require technical explanation
9. The second priority after PDFs is checking from the web-based agent tool

You must follow this exact format:

Question: the input question you must answer
Thought: your reasoning about what to do next (be thorough in your thinking)
Action: the tool name to use (must be one of: {tool_names}) - Do not use the youtube tool if it is not reqired in the question
Action Input: the input to pass to the tool
Observation: the result from the tool
... (you can repeat the Thought/Action/Action Input/Observation steps multiple times)
Thought: your final reasoning - synthesize what you've learned and organize your response
Final Answer: your comprehensive educational response that includes:
  - Clear explanation of concepts
  - Examples when helpful
  - Citations to course materials when applicable
  - Summary of key points
  - Recommended YouTube videos for further learning ONLY for substantive, educational queries that require in-depth explanation

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



st.title('Blue - Your Studdy Buddy')
st.markdown("By: *Rickmwasofficial* - **For Educational Purposes only**")

# Update the Streamlit app to display YouTube videos
def display_youtube_videos(videos):
    """Display YouTube video embeds in Streamlit."""
    if videos:
        st.markdown("#### Recommended YouTube Videos")
        cols = st.columns(len(videos))
        for i, video in enumerate(videos):
            with cols[i]:
                st.video(video['embed_link'])
                st.write(video['title'])

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
for messages in st.session_state.messages:
    if messages['role'] == "user": 
        message(messages['content'], is_user=True)
    else:
        message(messages['content'])

# React to user input
# Modify the response handling to include YouTube videos
if prompt := st.chat_input('Where is this unit taught?'):
    # Display user message in chat message container
    message(prompt, is_user=True)
    # Add user message to chat history
    st.session_state.messages.append({'role': "user", "content": prompt})

    response = agent_with_chat_history.invoke(
                {"input": prompt, "history": st.session_state.memory}, 
                {"configurable": {"session_id": "default"}}
            )
    
    # Search for YouTube videos related to the query
    try:
        youtube_videos = youtube_tool.run(prompt)
    except Exception as e:
        youtube_videos = []

    # Create a structured response
    processed_response = {
        'type': 'normal',
        'data': response['output']
    }

    print(st.session_state.memory)
    
    if response.get("intermediate_steps"):
        steps_markdown = ""
        for step in response["intermediate_steps"]:
            if isinstance(step, tuple) and len(step) > 0:
                agent_action = step[0]
                if hasattr(agent_action, "log"):
                    thought_log = agent_action.log
                    if "Thought:" in thought_log:
                        steps_markdown += f"- {thought_log.split('Action:')[0].strip()}\n"
        
        st.markdown(steps_markdown)
        
    message(processed_response['data'], allow_html=True)
    
    # Display YouTube videos if available
    if youtube_videos:
        display_youtube_videos(youtube_videos)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['output']})