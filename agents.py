from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# Load variables from .env file
load_dotenv()


# ============================================================
# MODEL SETUP
# ============================================================

# Groq LLM used by all agents and chains
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# 1st AGENT - SEARCH AGENT

# Search Agent → only searches the web using web_search
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a web research agent. "
            "Your job is ONLY to search the web and return useful search results. "
            "Use ONLY the web_search tool provided to you. "
            "Never call or invent tools such as web_open, open_url, browse, "
            "read_url, or any other tool. "
            "The web_search tool accepts exactly one argument named 'query'. "
            "Always provide the query argument as a string. "
            "Do not attempt to open or scrape URLs. "
            "Return the search results with titles, URLs and useful snippets."
        )
    )

# ============================================================
# 2nd AGENT - READER AGENT
# ============================================================

# Reader Agent → uses scrape_url tool
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


# ============================================================
# WRITER CHAIN
# ============================================================

# Prompt → Groq LLM → String output
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. "
        "Write clear, structured, factual and insightful reports."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:

- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional.
Do not invent information that is not present in the research."""
    ),
])

# Create Writer chain
writer_chain = writer_prompt | llm | StrOutputParser()


# ============================================================
# CRITIC CHAIN
# ============================================================

# Prompt → Groq LLM → String output
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. "
        "Be honest, specific and objective."
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""
    ),
])

# Create Critic chain
critic_chain = critic_prompt | llm | StrOutputParser()