import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# ==========================================
# INITIALIZE GEMINI LLM FOR CREWAI
# ==========================================
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.2
)

# ==========================================
# TOOL DEFINITIONS
# ==========================================

@tool("Tavily Startup Search")
def tavily_startup_search(query: str) -> str:
    """Searches the web for startups in a specific industry using Tavily."""
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    response = client.search(query=f"top innovative startups in {query} 2026 website", max_results=4)
    results = [f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}\n" for r in response.get("results", [])]
    return "\n---\n".join(results)

@tool("Website Scraper")
def scraper_tool(url: str) -> str:
    """Visits a startup website URL and extracts primary text content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ')
        return text[:3000] # Limit characters for the model
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# ==========================================
# AGENT DEFINITIONS (Powered by Gemini)
# ==========================================

researcher = Agent(
    role="Startup Market Researcher",
    goal="Find active startups in the given industry and collect their website links.",
    backstory="Expert at discovering emerging tech companies online.",
    tools=[tavily_startup_search],
    llm=gemini_llm,
    verbose=True
)

analyst = Agent(
    role="Startup Venture Analyst",
    goal="Visit startup websites, extract product insights, and identify funding stages.",
    backstory="Skilled at evaluating business models and offerings from company landing pages.",
    tools=[scraper_tool],
    llm=gemini_llm,
    verbose=True
)

writer = Agent(
    role="Investment Report Writer",
    goal="Create a final, well-structured Markdown startup report.",
    backstory="Professional tech journalist who writes clear, executive-ready reports.",
    llm=gemini_llm,
    verbose=True
)

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_startup_crew(industry: str):
    task1 = Task(
        description=f"Find 3 innovative startups in the {industry} industry with their website URLs.",
        expected_output="A list of startup names and absolute website links.",
        agent=researcher
    )
    
    task2 = Task(
        description="Visit each startup website using the scraper tool. Extract product summaries and identify their funding stage.",
        expected_output="Detailed notes on each company's product and funding stage.",
        agent=analyst
    )
    
    task3 = Task(
        description=f"Compile everything into a clean Markdown report for the {industry} industry ecosystem.",
        expected_output="A professional Markdown report.",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff(inputs={"industry": industry})

if __name__ == "__main__":
    target_industry = "Generative AI Healthcare"
    print(f"Starting Gemini Startup Research Team for: {target_industry}\n")
    report = run_startup_crew(target_industry)
    print("\n\n################# FINAL GEMINI REPORT #################\n")
    print(report)