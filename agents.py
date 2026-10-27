
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url

load_dotenv()

# Groq free-tier model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
)


# 1. Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a web research agent. "
            "Search for recent, reliable and relevant information. "
            "Use only the web_search tool for searching. "
            "Never invent URLs, sources or tool results. "
            "Return titles, URLs and useful snippets."
        ),
    )


# 2. Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a research reader agent. "
            "Identify the most relevant URL in the supplied search results. "
            "Call scrape_url with the exact URL. "
            "Summarize the extracted information accurately. "
            "Never invent scraped content."
        ),
    )


# 3. Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. "
        "Write clear, structured, factual and insightful reports."
    ),
    (
        "human",
        """Write a detailed research report.

Topic:
{topic}

Research Gathered:
{research}

Structure:
1. Introduction
2. Key Findings (at least 3 explained points)
3. Conclusion
4. Sources (URLs available in the research)

Be detailed and professional.
Do not invent facts or sources.
Distinguish verified findings from uncertain information."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# 4. Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict, constructive and objective research critic."
    ),
    (
        "human",
        """Evaluate the research report.

Report:
{report}

Use this exact format:

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

critic_chain = critic_prompt | llm | StrOutputParser()
