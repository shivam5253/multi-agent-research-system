
import time
from groq import RateLimitError
from langchain_core.messages import HumanMessage

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)


def invoke_with_retry(operation, attempts=3):
    """Retry temporary Groq rate-limit errors."""
    for attempt in range(attempts):
        try:
            return operation()
        except RateLimitError:
            if attempt == attempts - 1:
                raise

            wait_seconds = 5 * (attempt + 1)
            print(f"Groq rate limit reached. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # STEP 1: Search Agent
    print("\n" + "=" * 50)
    print("STEP 1 - Search Agent")
    print("=" * 50)

    search_agent = build_search_agent()

    search_result = invoke_with_retry(
        lambda: search_agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Find recent, reliable and detailed information "
                        f"about: {topic}"
                    )
                )
            ]
        })
    )

    state["search_results"] = (
        search_result["messages"][-1].content
    )

    # STEP 2: Reader Agent
    print("\n" + "=" * 50)
    print("STEP 2 - Reader Agent")
    print("=" * 50)

    reader_agent = build_reader_agent()

    reader_result = invoke_with_retry(
        lambda: reader_agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        f"Research topic: {topic}\n\n"
                        "Select the most relevant URL from these "
                        "search results and scrape it using the "
                        "scrape_url tool. Summarize the extracted "
                        "content accurately.\n\n"
                        f"Search results:\n"
                        f"{state['search_results'][:2500]}"
                    )
                )
            ]
        })
    )

    state["scraped_content"] = (
        reader_result["messages"][-1].content
    )

    # STEP 3: Writer Chain
    print("\n" + "=" * 50)
    print("STEP 3 - Writer Chain")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results'][:2500]}\n\n"
        f"DETAILED SCRAPED CONTENT:\n"
        f"{state['scraped_content'][:6000]}"
    )

    state["report"] = invoke_with_retry(
        lambda: writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
    )

    # STEP 4: Critic Chain
    print("\n" + "=" * 50)
    print("STEP 4 - Critic Chain")
    print("=" * 50)

    state["feedback"] = invoke_with_retry(
        lambda: critic_chain.invoke({
            "report": state["report"][:5000]
        })
    )

    print("\nResearch pipeline completed successfully.")

    return state


if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()

    if topic:
        result = run_research_pipeline(topic)
        print("\nFinal Report:\n", result["report"])
        print("\nCritic Feedback:\n", result["feedback"])
    else:
        print("Please enter a research topic.")

