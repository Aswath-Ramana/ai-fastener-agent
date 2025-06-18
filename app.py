import streamlit as st
import pandas as pd
import faiss
import os
import io
import time
import openai
from dotenv import load_dotenv

# --- Local Matcher Imports (for the bulk processing part) ---
from matcher import embed_text, search_top_k, fuzzy_match

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Fastener Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# --- Security & Prerequisite Check ---
if "OPENAI_API_KEY" not in os.environ:
    st.error("🚨 OpenAI API key is not set! Please create a .env file or set it in your deployment secrets.")
    st.info("The .env file should contain one line: OPENAI_API_KEY='sk-xxxxxxxxxx'")
    st.stop()
else:
    # Initialize the OpenAI client globally
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- AGENT CONFIGURATION ---
# Your specific Assistant ID is now hardcoded.
ASSISTANT_ID = "asst_gUAK10UlLZY6nZR2ye4a9fc6"

# --- Caching Functions for Bulk Matching ---
@st.cache_resource
def load_local_search_components():
    """Loads components needed ONLY for the local candidate search in bulk mode."""
    st.info("Loading local search components for bulk processing...")
    try:
        index = faiss.read_index("local_search_assets/faiss_index.bin")
        master_df = pd.read_parquet("local_search_assets/master_metadata.parquet")
        st.success("✅ Local search components loaded.")
        return index, master_df
    except Exception as e:
        st.error(f"🚨 Failed to load local search components: {e}")
        st.error("Please ensure the 'local_search_assets' folder with index files exists.")
        return None, None

def dataframe_to_excel_bytes(df):
    """Converts a DataFrame to an in-memory Excel file (bytes)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Matched_Orders')
    processed_data = output.getvalue()
    return processed_data

# --- Main App UI ---
st.title("🔩 AI Fastener Agent")
st.markdown("Use the **Automated Order Matcher** for bulk processing or chat with the **Fastener Finder Assistant** for specific queries.")

tab1, tab2 = st.tabs(["🤖 Automated Order Matcher", "💬 Fastener Finder Assistant (Chat)"])

# ==============================================================================
# TAB 1: AUTOMATED ORDER MATCHER (HYBRID WORKFLOW)
# ==============================================================================
with tab1:
    st.header("Upload an Order File for AI-Assisted Matching")
    st.info("This process uses local search to find 5 candidates, then asks the AI Assistant to make the final decision for each row.")
    
    (local_index, master_df) = load_local_search_components()

    if local_index is None:
        st.stop()

    order_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"], key="bulk_uploader")

    if order_file:
        st.subheader("Raw Order Data Preview")
        order_df = pd.read_excel(order_file)
        st.dataframe(order_df.head())

        # --- Column Mapping ---
        def find_column(keywords, columns):
            for col in columns:
                if any(k.lower() in col.lower() for k in keywords):
                    return col
            return None

        with st.expander("⚙️ Step 1: Map Your Columns", expanded=True):
            col1, col2, col3 = st.columns(3)
            part_col = col1.selectbox("Part Number Column", options=order_df.columns, index=order_df.columns.get_loc(find_column(['part', 'item'], order_df.columns)) if find_column(['part', 'item'], order_df.columns) else 0)
            desc_col = col2.selectbox("Description Column", options=order_df.columns, index=order_df.columns.get_loc(find_column(['desc'], order_df.columns)) if find_column(['desc'], order_df.columns) else 1)
            dim_col = col3.selectbox("Dimension Column (Optional)", options=[None] + list(order_df.columns), index=0)

        # --- Main Processing Logic ---
        if st.button("🚀 Start Hybrid Matching", type="primary"):
            with st.spinner("AI Agent is analyzing your file... This may take a while."):
                if dim_col:
                    order_df['query_text'] = order_df[[desc_col, dim_col]].fillna('').astype(str).agg(' '.join, axis=1)
                else:
                    order_df['query_text'] = order_df[desc_col].fillna('').astype(str)

                results = []
                progress_bar = st.progress(0, "Initializing...")
                total_rows = len(order_df)

                for i, row in order_df.iterrows():
                    query = row['query_text']
                    progress_text = f"Processing '{row[part_col]}'... ({i+1}/{total_rows})"
                    progress_bar.progress((i + 1) / total_rows, progress_text)

                    # 1. Local Candidate Sourcing
                    query_vec = embed_text([query])[0]
                    top_indices, _ = search_top_k(local_index, query_vec, k=3)
                    semantic_matches = master_df.iloc[top_indices]
                    
                    fuzz_matches = fuzzy_match(query, master_df['Sales-Description'].tolist(), limit=2)
                    fuzz_indices = [match[2] for match in fuzz_matches]
                    fuzzy_matches = master_df.iloc[fuzz_indices]
                    
                    all_candidates = pd.concat([semantic_matches, fuzzy_matches]).drop_duplicates(subset=['Item']).reset_index(drop=True)

                    if all_candidates.empty:
                        results.append({
                            "Original_Part#": row[part_col], "Original_Description": row[desc_col],
                            "Original_Dimension": row[dim_col] if dim_col else "",
                            "Matched_Item": "New Item", "Matched_Sales-Description": "", "AI_Decision_Raw": "No local candidates found"
                        })
                        continue

                    # 2. Package the Problem for the Assistant
                    candidates_str = all_candidates[['Item', 'Sales-Description', 'Dimension (complete)']].to_string(index=False)
                    
                    system_instruction = """
                    You are an AI decision-making agent. Your task is to analyze a customer query and a list of up to 5 potential product matches that have been pre-selected for you.
                    Based on your internal knowledge base (which includes detailed specs), you must select the SINGLE best match.
                    
                    **Your response MUST be ONLY the `Item Number` of the best match.**
                    
                    If, after reviewing your internal knowledge, you determine that NONE of the candidates are a suitable match for the customer's query, you MUST respond with the exact phrase: `New Item`.
                    Do not provide any explanation or extra text. Just the Item Number or 'New Item'.
                    """
                    
                    user_prompt = f"""
                    Customer Query: "{query}"
                    
                    Pre-selected Candidates:
                    {candidates_str}
                    
                    Please analyze these candidates against your knowledge base and return the Item Number of the best match, or 'New Item'.
                    """

                    # 3. One-Shot Assistant Call
                    try:
                        thread = client.beta.threads.create()
                        client.beta.threads.messages.create(
                            thread_id=thread.id, role="user", content=user_prompt
                        )
                        run = client.beta.threads.runs.create(
                            thread_id=thread.id, assistant_id=ASSISTANT_ID, instructions=system_instruction
                        )

                        while run.status in ['queued', 'in_progress']:
                            time.sleep(0.5)
                            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                        
                        messages = client.beta.threads.messages.list(thread_id=thread.id)
                        ai_response = messages.data[0].content[0].text.value.strip()

                    except Exception as e:
                        ai_response = f"Error: {e}"
                        st.warning(f"An error occurred for item {row[part_col]}: {e}")

                    # 4. Compile the Final Result
                    final_row = {
                        "Original_Part#": row[part_col],
                        "Original_Description": row[desc_col],
                        "Original_Dimension": row[dim_col] if dim_col else "",
                        "Matched_Item": "New Item",
                        "Matched_Sales-Description": "",
                        "AI_Decision_Raw": ai_response
                    }

                    if ai_response != "New Item" and not ai_response.startswith("Error"):
                        try:
                            matched_item_data_df = master_df[master_df['Item'] == ai_response]
                            if not matched_item_data_df.empty:
                                matched_item_data = matched_item_data_df.iloc[0]
                                final_row.update({
                                    "Matched_Item": matched_item_data['Item'],
                                    "Matched_Sales-Description": matched_item_data['Sales-Description'],
                                })
                        except Exception:
                             # If lookup fails, it remains a "New Item"
                            final_row["AI_Decision_Raw"] += " (Failed to look up in master_df)"

                    results.append(final_row)

                st.session_state.results_df = pd.DataFrame(results)

            st.success("✅ Hybrid matching complete!")

    if 'results_df' in st.session_state:
        st.header("📊 Final Match Results")
        st.dataframe(st.session_state.results_df)

        excel_data = dataframe_to_excel_bytes(st.session_state.results_df)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_data,
            file_name="ai_matched_orders.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ==============================================================================
# TAB 2: FASTENER FINDER ASSISTANT (CHATBOT)
# ==============================================================================
with tab2:
    st.header("Chat with the Fastener Finder Assistant")
    st.info("This AI agent uses its internal knowledge base (Master Data & Catalog) to answer your questions.")

    if "start_chat" not in st.session_state:
        st.session_state.start_chat = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    if st.button("Start New Chat Session"):
        st.session_state.start_chat = True
        with st.spinner("Initializing a new conversation thread..."):
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id
        st.session_state.agent_messages = []
        st.success("New chat session started.")
        st.rerun()

    if st.session_state.start_chat:
        for message in st.session_state.agent_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about a product, e.g., 'Do you have M8 hex bolts in stainless steel?'"):
            st.session_state.agent_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt
            )

            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=ASSISTANT_ID
            )

            with st.spinner("The AI agent is thinking... This may take a moment."):
                while run.status in ['queued', 'in_progress']:
                    time.sleep(1)
                    run = client.beta.threads.runs.retrieve(
                        thread_id=st.session_state.thread_id,
                        run_id=run.id
                    )

            messages = client.beta.threads.messages.list(
                thread_id=st.session_state.thread_id
            )
            
            assistant_message = messages.data[0]
            response_text = ""
            for content_part in assistant_message.content:
                if content_part.type == 'text':
                    response_text += content_part.text.value

            with st.chat_message("assistant"):
                st.markdown(response_text)
            st.session_state.agent_messages.append({"role": "assistant", "content": response_text})