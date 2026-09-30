from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from rag_1 import ingest_data

load_dotenv()

print("All libraries imported")

# Tool 1 : web search tool
web_search = DuckDuckGoSearchRun()


# Tool 2 : Lab report data retriever tool
@tool
def get_lab_report_data(query: str) -> str:
    """Retrieve chunks of the patient's lab report
    via similarity search (e.g. query='flagged results',
    query='glucose', query='cholesterol')."""
    vector_store = ingest_data()
    response = vector_store.similarity_search(query)
    return response


if __name__ == "__main__":
    # invoke web search tool
    # response = web_search.invoke("Olympics 2028")
    # print(response)

    print(get_lab_report_data.invoke({"query": "flagged results"})[0])
    print(f"Tool Name : {get_lab_report_data.name}")
    print(f"Tool Arguments : {get_lab_report_data.args}")
    print(f"Tool Description : {get_lab_report_data.description}")