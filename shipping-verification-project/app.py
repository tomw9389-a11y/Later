import os
import json
import streamlit as st
import requests

st.set_page_config(page_title="Shipping Document Verification Dashboard", layout="wide")

# App Header
st.title("🚢 Automated Shipping Document Verification")
st.markdown("AI-powered pipeline for email classification, BL/SI extraction, and automated discrepancy detection.")

# Sidebar Controls
st.sidebar.header("Pipeline Control Panel")
data_dir = "data"

if st.sidebar.button("Run Full Pipeline (Background)"):
    with st.spinner("Processing inbox... This may take a while depending on API limits."):
        # Import and run your verification system
        from solution import DocumentVerificationSystem
        system = DocumentVerificationSystem(data_dir=data_dir)
        system.process_inbox()
    st.sidebar.success("Processing complete!")

# Load Results JSON if it exists
output_file = "sample_submission.json"
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        results = json.load(f)
    
    # Metrics Row
    total_emails = len(results)
    doc_comparisons = sum(1 for r in results.values() if r.get("category") == "document_comparison")
    mismatches_found = sum(1 for r in results.values() if r.get("mismatch_found"))
    human_reviews = sum(1 for r in results.values() if r.get("needs_human_review"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Processed", total_emails)
    col2.metric("Document Comparisons", doc_comparisons)
    col3.metric("Mismatches Flagged", mismatches_found, delta_color="inverse")
    col4.metric("Human Review Needed", human_reviews)

    st.markdown("---")

    # Filter & Search Table
    st.subheader("Inbox Audit Trail")
    search_query = st.text_input("Search by Email ID (e.g., email_110):")

    for email_id, record in results.items():
        if search_query and search_query.lower() not in email_id.lower():
            continue
            
        category = record.get("category")
        mismatch = record.get("mismatch_found")
        review = record.get("needs_human_review")
        
        # Color code the expander based on status
        status_icon = "❌" if mismatch else ("⚠️" if review else "✅")
        
        with st.expander(f"{status_icon} {email_id} — Category: **{category}**"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Status Flags:**")
                st.json({
                    "needs_human_review": review,
                    "mismatch_found": mismatch
                })
            with c2:
                st.write("**Discrepancies Details:**")
                st.json(record.get("discrepancies", {}))

    # Docker Evaluation Trigger
    st.markdown("---")
    st.subheader("Docker Evaluation Server")
    if st.button("Submit to Local Docker Scoreboard"):
        try:
            with open(output_file, "r") as f:
                payload = json.load(f)
            response = requests.post("http://localhost:8080/submit", json=payload)
            if response.status_code == 200:
                st.success("Successfully submitted to Docker evaluation server!")
                st.json(response.json())
            else:
                st.error(f"Server responded with status {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to Docker server: {e}")

else:
    st.info("No `sample_submission.json` found yet. Click the button in the sidebar or run your python script to generate results.")