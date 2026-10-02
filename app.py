import streamlit as st

from pipeline import run_research_pipeline


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔎 Multi-Agent Research System")

st.write(
    "AI-powered research using Search, Reader, Writer and Critic agents."
)


# ============================================================
# USER INPUT
# ============================================================

st.subheader("📝 Research Topic")

topic = st.text_input(
    "Enter your research topic",
    placeholder="Example: Latest developments in Artificial Intelligence"
)


# ============================================================
# START RESEARCH
# ============================================================

if st.button("🚀 Start Research", use_container_width=True):

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        with st.spinner("🤖 Multi-agent system is researching..."):

            result = run_research_pipeline(topic)


        st.success("✅ Research completed successfully!")


        # ====================================================
        # RESULTS
        # ====================================================

        tab1, tab2, tab3, tab4 = st.tabs([
            "🔎 Search Results",
            "📖 Research",
            "📝 Final Report",
            "🧐 Critic Feedback"
        ])


        # ====================================================
        # SEARCH RESULTS
        # ====================================================

        with tab1:

            st.subheader("🔎 Search Results")

            st.markdown(
                result["search_results"]
            )


        # ====================================================
        # SCRAPED CONTENT
        # ====================================================

        with tab2:

            st.subheader("📖 Detailed Research")

            st.markdown(
                result["scraped_content"]
            )


        # ====================================================
        # FINAL REPORT
        # ====================================================

        with tab3:

            st.subheader("📝 Final Research Report")

            st.markdown(
                result["report"]
            )


        # ====================================================
        # CRITIC FEEDBACK
        # ====================================================

        with tab4:

            st.subheader("🧐 Critic Feedback")

            st.markdown(
                result["feedback"]
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with LangChain • Groq • Tavily • BeautifulSoup • Streamlit"
)