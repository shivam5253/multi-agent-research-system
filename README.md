# 🔎 Multi-Agent Research System

An AI-powered multi-agent research application built with **Python, LangChain, Groq, Tavily, BeautifulSoup, and Streamlit**.

The system takes a research topic from the user and passes it through multiple specialized stages:

**Search → Reader → Writer → Critic**

The final result is presented through a simple Streamlit web interface.

---


---

## 📌 Project Overview

The Multi-Agent Research System automates the process of researching a topic from the web and generating a structured research report.

Instead of using a single LLM prompt for everything, the application separates the work into specialized agents and chains.

### Workflow

```text
                    User
                     │
                     ▼
              Research Topic
                     │
                     ▼
            ┌─────────────────┐
            │  Search Agent   │
            │                 │
            │ Tavily Web      │
            │ Search          │
            └────────┬────────┘
                     │
                     ▼
              Search Results
                     │
                     ▼
            ┌─────────────────┐
            │  Reader Agent   │
            │                 │
            │ URL Selection + │
            │ Web Scraping    │
            └────────┬────────┘
                     │
                     ▼
             Detailed Content
                     │
                     ▼
            ┌─────────────────┐
            │  Writer Chain   │
            │                 │
            │ Research Report │
            └────────┬────────┘
                     │
                     ▼
              Final Report
                     │
                     ▼
            ┌─────────────────┐
            │  Critic Chain   │
            │                 │
            │ Review + Score  │
            └────────┬────────┘
                     │
                     ▼
              Critic Feedback

## 🚀 Live Demo

https://multi-agent-research-system-9bbqdpm3nmchagajzbhqez.streamlit.app/
