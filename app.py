import os
import time
import pandas as pd
import streamlit as st
import altair as alt

from retrieval import HybridRetriever, SearchStrategy
from evaluation import BenchmarkEngine
from config import settings

st.set_page_config(
    page_title="Legal RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .stat-box {padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_system():
    return HybridRetriever()

def render_search_ui(engine):
    st.header("Contract Clause Search")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Legal Query", placeholder="e.g., What is the termination notice period?")
    with col2:
        strategy = st.selectbox(
            "Strategy", 
            [s.value for s in SearchStrategy], 
            index=2,
            format_func=lambda x: x.replace("_", " ").upper()
        )
    with col3:
        top_k = st.number_input("Top K", min_value=1, max_value=10, value=5)

    if st.button("Search Contracts", type="primary"):
        if not query:
            st.warning("Please enter a query.")
            return

        start = time.time()
        with st.spinner(f"Running {strategy} pipeline..."):
            try:
                results = engine.search(query, strategy=SearchStrategy(strategy))
                duration = time.time() - start
                
                st.success(f"Found {len(results)} relevant clauses in {duration:.3f}s")
                
                for res in results[:top_k]:
                    with st.container():
                        title = res.metadata.get('title', 'Unknown Contract')
                        st.markdown(f"### {title}")
                        st.markdown(f"> {res.content}")
                        
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        meta_col1.caption(f"**Score:** {res.score:.4f}")
                        source_display = getattr(res, 'source_strategy', 'Hybrid').upper()
                        meta_col2.caption(f"**Source:** {source_display}")
                        doc_id = res.metadata.get('doc_id', 'N/A')
                        meta_col3.caption(f"**Doc ID:** {doc_id}")
                        st.divider()
                        
            except Exception as e:
                st.error(f"Search Failed: {str(e)}")

def render_admin_ui():
    st.header("Logs")
    
    tab1, tab2 = st.tabs(["Ingestion Log", "Query Log"])
    
    with tab1:
        if os.path.exists(settings.INGESTION_AUDIT_FILE):
            df = pd.read_csv(settings.INGESTION_AUDIT_FILE)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Docs Processed", len(df))
            
            success_count = len(df[df['status']=='SUCCESS']) + len(df[df['status']=='STAGED'])
            success_rate = (success_count/len(df))*100 if len(df) > 0 else 0
            
            m2.metric("Success Rate", f"{success_rate:.1f}%")
            m3.metric("Avg Char Length", f"{int(df['clean_len'].mean())}")
            m4.metric("Last Ingest", df['timestamp'].max())
            
            st.dataframe(
                df.sort_values("timestamp", ascending=False),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No ingestion logs found. Run ETL pipeline first.")

    with tab2:
        if os.path.exists(settings.QUERY_LOG_FILE):
            df = pd.read_csv(settings.QUERY_LOG_FILE)
            
            chart = alt.Chart(df).mark_line(point=True).encode(
                x='timestamp',
                y='total_latency_ms',
                tooltip=['strategy', 'total_latency_ms']
            ).properties(title="Query Latency Over Time")
            
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
        else:
            st.info("No query logs found. Perform some searches.")

def render_lab_ui():
    st.header("A/B Test Results")
    st.markdown("Run the golden set queries, calculate MRR & NDCG, and regenerate the assets.")
    
    if st.button("Run Benchmark"):
        bencher = BenchmarkEngine()
        
        status_text = st.empty()
        status_text.text("Running Benchmark... Check terminal for details.")
        
        bencher.run_ab_test()
        status_text.success("Done. Assets updated in /assets.")
        
        asset_path = os.path.join(settings.ASSETS_DIR, "benchmark_comparison.png")
        if os.path.exists(asset_path):
            st.image(asset_path, caption="Live Benchmark Results", use_container_width=False)
        else:
            st.warning("Benchmark ran, but image generation failed.")

def main():
    st.sidebar.title("Legal RAG")
    st.sidebar.image("https://img.icons8.com/color/96/law.png", width=50)
    page = st.sidebar.radio("Navigation", ["Search", "Admin Console", "Research Lab"])
    
    st.sidebar.divider()
    st.sidebar.info(f"Environment: {settings.ENV}\nDevice: {settings.EMBEDDING_DEVICE}")

    with st.spinner("Loading models..."):
        engine = load_system()

    if page == "Search":
        render_search_ui(engine)
    elif page == "Admin Console":
        render_admin_ui()
    elif page == "Research Lab":
        render_lab_ui()

if __name__ == "__main__":
    main()