
import time
from groq import RateLimitError

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # STEP 1: Search Agent
    print("\n" + " =" * 50)
    print("Step 1 - Search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Find recent, reliable and detailed information about: {topic}"
            )
        ]
    })

    state["search_results"] = search_result["messages"][-1].content

    print("\nSearch result:\n", state["search_results"])

    # STEP 2: Reader Agent
    print("\n" + " =" * 50)
    print("Step 2 - Reader agent is scraping top resources...")
    print("=" * 50)

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"Based on the following search results about '{topic}', "
                "pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:2500]}"
            )
        ]
    })

    state["scraped_content"] = reader_result["messages"][-1].content

    print("\nScraped content:\n", state["scraped_content"])

    # STEP 3: Writer Agent
    print("\n" + " =" * 50)
    print("Step 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results'][:2500]}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content'][:6000]}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\nFinal Report:\n", state["report"])

    # STEP 4: Critic Agent
    print("\n" + " =" * 50)
    print("Step 4 - Critic is reviewing the report...")
    print("=" * 50)

    # Limit the report sent to the critic
    report_for_critic = str(state["report"])[:5000]

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            state["feedback"] = critic_chain.invoke({
                "report": report_for_critic
            })
            break

        except RateLimitError:
            if attempt == max_attempts - 1:
                print("Critic failed after multiple rate-limit retries.")
                raise

            wait_seconds = 5 * (attempt + 1)

            print(
                f"Groq rate limit reached. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    print("\nCritic feedback:\n", state["feedback"])

    # Return all pipeline results to the Streamlit app
    return state


# Run pipeline when this file is executed directly
if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    result = run_research_pipeline(topic)
