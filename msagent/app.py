from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from googleapiclient.discovery import build
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

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")

# FastAPI initialization
app = FastAPI()

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
  - Recommended YouTube videos for further learning ONLY for substantive, educational queries that require in-depth explanation.

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

# Define the API's data model for user input
class UserQuery(BaseModel):
    query: str

@app.post("/ask/")
async def ask_question(user_query: UserQuery):
    """Handles user questions and returns responses from the agent."""
    try:
        response = executor.invoke({"input": user_query.query})
        return {"response": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def search_youtube_videos(query):
    """Search for relevant YouTube videos."""
    youtube = build('youtube', 'v3', developerKey=os.getenv("GOOGLE_API_KEY"))
    request = youtube.search().list(
        part="snippet",
        maxResults=3,
        q=query + " tutorial",
        type="video",
        videoEmbeddable="true",
    )
    response = request.execute()
    
    # Return a list of YouTube video data
    videos = []
    for item in response['items']:
        videos.append({
            'key': item['id']['videoId'],  # Return the videoId as the key
            'title': item['snippet']['title'],
            'embed_link': f"https://www.youtube.com/embed/{item['id']['videoId']}"
        })
    
    return videos

# Define the request body model
class UserQuery(BaseModel):
    query: str

# Endpoint to search YouTube
@app.post("/youtube/")
async def search_youtube(query: UserQuery):
    """Returns educational YouTube videos based on the query."""
    try:
        videos = search_youtube_videos(query.query)  # Use the 'query' from UserQuery
        return {"videos": videos}  # Return the video details with 'key' instead of 'id'
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))