from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

def create_llm():
    llm=ChatGroq(model="openai/gpt-oss-20b")
    return llm
