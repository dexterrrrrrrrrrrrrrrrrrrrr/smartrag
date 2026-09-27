"""
SmartRAG Streamlit Dashboard.

Run with:
    streamlit run dashboard/app.py

Talks to the FastAPI backend over HTTP only (see api_client.py) — start the
backend first with `python run.py`.
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiClient

API_BASE_URL = os.environ.get("SMARTRAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="SmartRAG Dashboard", page_icon="🧠", layout="wide")

client = ApiClient(API_BASE_URL)


def safe_call(fn, *args, **kwargs):
    """Call the API and show a friendly error instead of crashing the
    dashboard if the backend or one of its dependencies is down."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ----------------------------------------------------------------------------
# Sidebar: health + cache controls
# ----------------------------------------------------------------------------
st.sidebar.title("🧠 SmartRAG")
st.sidebar.caption("Semantic Caching & Cost-Aware LLM Routing")

health, health_err = safe_call(client.health)
if health_err:
    st.sidebar.error(f"Backend unreachable: {health_err}")
else:
    icon = "🟢" if health["status"] == "healthy" else "🟡"
    st.sidebar.markdown(f"{icon} **Status: {health['status'].upper()}**")
    st.sidebar.markdown(f"- Ollama: {'✅' if health['ollama_available'] else '❌'}")
    st.sidebar.markdown(f"- Qdrant: {'✅' if health['qdrant_available'] else '❌'}")
    st.sidebar.markdown(f"- Redis: {'✅' if health['redis_available'] else '❌'}")

st.sidebar.divider()
st.sidebar.subheader("Semantic Cache")
cache_stats, _ = safe_call(client.cache_stats)
if cache_stats:
    new_threshold = st.sidebar.slider(
        "Similarity threshold",
        min_value=0.5,
        max_value=1.0,
        value=float(cache_stats["similarity_threshold"]),
        step=0.01,
        help="Minimum cosine similarity for a cache hit.",
    )
    if st.sidebar.button("Apply threshold"):
        safe_call(client.set_cache_threshold, new_threshold)
        st.sidebar.success(f"Threshold set to {new_threshold}")
    st.sidebar.metric("Cached entries", cache_stats["num_entries"])
    if st.sidebar.button("🗑️ Clear cache"):
        result, err = safe_call(client.clear_cache)
        if result:
            st.sidebar.success(f"Cleared {result['entries_cleared']} entries")

if st.sidebar.button("🗑️ Clear all analytics"):
    result, err = safe_call(client.clear_analytics)
    if result:
        st.sidebar.success(f"Cleared {result['rows_cleared']} log rows")

tab_chat, tab_docs, tab_overview, tab_cache, tab_routing, tab_perf = st.tabs(
    ["💬 Chat", "📄 Documents", "📊 Overview", "💾 Cache Analytics", "🧭 Model Routing", "⚡ Performance"]
)

# ----------------------------------------------------------------------------
# Chat tab
# ----------------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask a question about your uploaded documents")
    query_text = st.text_input("Your question", key="query_input")
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and query_text.strip():
        with st.spinner("Thinking..."):
            result, err = safe_call(client.query, query_text)
        if err:
            st.error(f"Request failed: {err}")
        else:
            st.markdown("#### Answer")
            st.write(result["answer"])

            if result["sources"]:
                st.markdown("#### Sources")
                for s in result["sources"]:
                    page = f" — Page {s['page_number']}" if s.get("page_number") else ""
                    st.markdown(f"📄 {s['document_name']}{page}")

            badge = "💾 Cache HIT" if result["cache_hit"] else "🔄 Cache MISS"
            st.markdown(f"**{badge}**")
            if result["cache_hit"] and result["similarity_score"] is not None:
                st.caption(f"Similarity: {result['similarity_score']:.4f} — no LLM inference was needed.")

            with st.expander("Request details"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Total latency", f"{result['latency']['total_ms']:.0f} ms")
                col2.metric("Estimated cost", f"${result['estimated_cost_usd']:.6f}")
                if result["cache_hit"]:
                    avoided = result["estimated_cost_if_no_cache_usd"]
                    col3.metric("Cost avoided", f"${avoided:.6f}")
                elif result.get("route"):
                    col3.metric("Model", result["route"]["role"])
                if result.get("route"):
                    st.write(f"Complexity: **{result['route']['complexity_label']}**")
                    st.write(f"Reason: {result['route']['reason']}")
                st.json(result["latency"])

# ----------------------------------------------------------------------------
# Documents tab
# ----------------------------------------------------------------------------
with tab_docs:
    st.subheader("Upload documents (PDF, TXT, Markdown)")
    uploaded = st.file_uploader("Choose a file", type=["pdf", "txt", "md", "markdown"])
    if uploaded is not None and st.button("Ingest document"):
        with st.spinner(f"Ingesting {uploaded.name}..."):
            result, err = safe_call(client.upload_document, uploaded.name, uploaded.getvalue())
        if err:
            st.error(f"Ingestion failed: {err}")
        else:
            st.success(
                f"Ingested '{result['document_name']}': {result['num_chunks']} chunks, "
                f"{result['total_characters']} characters, {result['latency_ms']:.0f}ms"
            )

    count, err = safe_call(client.document_count)
    if count:
        st.metric("Chunks in knowledge base", count["chunk_count"])
    if st.button("🗑️ Clear knowledge base"):
        safe_call(client.clear_documents)
        st.success("Knowledge base cleared.")

# ----------------------------------------------------------------------------
# Overview tab
# ----------------------------------------------------------------------------
with tab_overview:
    st.subheader("System Overview")
    st.caption(
        "All figures below come from real requests handled by this running instance. "
        "Cost figures are **estimated / hypothetical API-equivalent cost** — this app "
        "uses free local models, so no real money is ever spent."
    )
    data, err = safe_call(client.analytics_overview)
    if err:
        st.error(err)
    elif data["total_queries"] == 0:
        st.info("No queries yet — ask something in the Chat tab to populate this dashboard.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries", data["total_queries"])
        c2.metric("Cache Hit Rate", f"{data['cache_hit_rate']*100:.1f}%" if data["cache_hit_rate"] is not None else "—")
        c3.metric("LLM Calls", data["llm_calls"])
        c4.metric("LLM Calls Avoided", data["llm_calls_avoided"])

        c5, c6, c7, c8 = st.columns(4)
        lat = data["latency"]
        c5.metric("Avg Latency", f"{lat['avg']:.0f} ms" if lat["avg"] is not None else "—")
        c6.metric("p95 Latency", f"{lat['p95']:.0f} ms" if lat["p95"] is not None else "—")
        c7.metric("Est. API-equiv. Cost", f"${data['estimated_cost_usd']:.6f}")
        c8.metric("Est. Cost Avoided", f"${data['estimated_cost_avoided_usd']:.6f}")

# ----------------------------------------------------------------------------
# Cache Analytics tab
# ----------------------------------------------------------------------------
with tab_cache:
    st.subheader("Cache Analytics")
    data, err = safe_call(client.analytics_cache)
    if err:
        st.error(err)
    elif data["cache_hits"] + data["cache_misses"] == 0:
        st.info("No queries yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                names=["Cache Hits", "Cache Misses"],
                values=[data["cache_hits"], data["cache_misses"]],
                title="Cache Hits vs Misses",
                color_discrete_sequence=["#22c55e", "#f97316"],
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if data["similarity_scores"]:
                fig2 = px.histogram(
                    x=data["similarity_scores"], nbins=20, title="Similarity Score Distribution"
                )
                fig2.update_layout(xaxis_title="Cosine similarity", yaxis_title="Count")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No cache hits yet to plot a similarity distribution.")

        lookup = data["cache_lookup_latency"]
        st.metric("Avg cache lookup latency", f"{lookup['avg']:.2f} ms" if lookup["avg"] is not None else "—")

# ----------------------------------------------------------------------------
# Model Routing tab
# ----------------------------------------------------------------------------
with tab_routing:
    st.subheader("Model Routing")
    data, err = safe_call(client.analytics_routing)
    if err:
        st.error(err)
    elif data["small_model_requests"] + data["large_model_requests"] == 0:
        st.info("No LLM calls yet (all queries were cache hits, or none have been asked).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                x=["SMALL_MODEL", "LARGE_MODEL"],
                y=[data["small_model_requests"], data["large_model_requests"]],
                title="Requests by Model",
                labels={"x": "Model", "y": "Requests"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(
                x=["SMALL_MODEL", "LARGE_MODEL"],
                y=[data["cost_by_model"]["small"], data["cost_by_model"]["large"]],
                title="Estimated Cost by Model (USD)",
                labels={"x": "Model", "y": "Estimated cost ($)"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(
            x=["LOW", "HIGH"],
            y=[data["complexity_distribution"]["LOW"], data["complexity_distribution"]["HIGH"]],
            title="Query Complexity Distribution",
            labels={"x": "Complexity", "y": "Queries"},
        )
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# Performance tab
# ----------------------------------------------------------------------------
with tab_perf:
    st.subheader("Performance")
    data, err = safe_call(client.analytics_performance)
    if err:
        st.error(err)
    elif not data["timeseries"]:
        st.info("No queries yet.")
    else:
        df = pd.DataFrame(data["timeseries"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fig = px.line(df, x="timestamp", y="total_latency_ms", title="Total Latency Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        for col, label, key in [
            (col1, "Total latency", "total_latency"),
            (col2, "Retrieval latency", "retrieval_latency"),
            (col3, "LLM latency", "llm_latency"),
        ]:
            stats = data[key]
            with col:
                st.markdown(f"**{label}**")
                st.write(f"avg: {stats['avg']} ms" if stats["avg"] is not None else "avg: —")
                st.write(f"p50: {stats['p50']} ms" if stats["p50"] is not None else "p50: —")
                st.write(f"p95: {stats['p95']} ms" if stats["p95"] is not None else "p95: —")
                st.write(f"p99: {stats['p99']} ms" if stats["p99"] is not None else "p99: — (needs 20+ samples)")
