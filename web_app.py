import os
import base64
import hashlib  # Used to calculate unique file fingerprints (MD5)
import streamlit as st
import pandas as pd
from typing import List, Union
from pydantic import BaseModel, Field

# Import AI SDKs
from google import genai
from google.genai import types
from openai import OpenAI

# =====================================================================
# 1. STRUCTURE BLUEPRINTS (Universal Data Schemas)
# =====================================================================

# --- 1A. Purchase Invoice Data Structures ---
class PurchaseLineItem(BaseModel):
    item_name: str = Field(description="The description or name of the product or service")
    quantity: float = Field(default=0.0, description="The quantity purchased")
    rate: float = Field(default=0.0, description="The unit price or rate per item")
    item_amount: float = Field(default=0.0, description="Total amount for this item before tax")
    cgst_amount: float = Field(default=0.0, description="CGST tax component amount")
    sgst_amount: float = Field(default=0.0, description="SGST tax component amount")
    igst_amount: float = Field(default=0.0, description="IGST tax component amount")

class PurchaseInvoiceData(BaseModel):
    supplier_name: str = Field(description="The company issuing the invoice (seller/supplier)")
    invoice_number: str = Field(description="The unique purchase invoice number")
    invoice_date: str = Field(description="The date issued (YYYY-MM-DD)")
    line_items: List[PurchaseLineItem] = Field(description="List of items billed")
    total_amount: float = Field(description="The final total amount payable")


# --- 1B. Bank Statement Data Structures ---
class BankTransactionRow(BaseModel):
    transaction_date: str = Field(description="The date of the transaction (YYYY-MM-DD format if possible)")
    particulars: str = Field(description="The description, particulars, or narration of the transaction")
    debit_amount: float = Field(default=0.0, description="Amount debited/withdrawn. Set to 0.0 if empty or a credit.")
    credit_amount: float = Field(default=0.0, description="Amount credited/deposited. Set to 0.0 if empty or a debit.")
    closing_balance: float = Field(default=0.0, description="The running balance after the transaction")

class BankStatementData(BaseModel):
    account_holder_name: str = Field(description="Name of the bank account holder")
    bank_name: str = Field(description="Name of the banking institution")
    transactions: List[BankTransactionRow] = Field(description="Chronological list of all transaction rows found in the statement")


# --- 1C. Sale Invoice Data Structures ---
class SaleLineItem(BaseModel):
    item_name: str = Field(description="The description or name of the product or service sold")
    quantity: float = Field(default=0.0, description="The quantity sold")
    rate: float = Field(default=0.0, description="The unit price or rate charged per item")
    item_amount: float = Field(default=0.0, description="Total amount for this item before tax")
    igst_amount: float = Field(default=0.0, description="IGST tax component amount")
    cgst_amount: float = Field(default=0.0, description="CGST tax component amount")
    sgst_amount: float = Field(default=0.0, description="SGST tax component amount")

class SaleInvoiceData(BaseModel):
    invoice_date: str = Field(description="The date the sale invoice was issued (YYYY-MM-DD)")
    invoice_number: str = Field(description="The unique sale invoice number")
    buyer_name: str = Field(description="The name of the customer or buyer receiving the invoice")
    buyer_gst_number: str = Field(default="", description="The GSTIN/GST number of the buyer or customer. Leave blank if not available.")
    line_items: List[SaleLineItem] = Field(description="List of items sold in this sale bill")
    total_amount: float = Field(description="The grand total amount of the sale invoice including all taxes")

# =====================================================================
# 2. UNIFIED AI PROCESSING ENGINES
# =====================================================================
def extract_with_gemini(uploaded_file, api_key: str, target_schema, system_instruction: str):
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client()
    
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    mime_type = "application/pdf" if ext == ".pdf" else f"image/{ext.replace('.', '')}"
    file_bytes = uploaded_file.getvalue()
        
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            system_instruction
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=target_schema,
            temperature=0.0,
        ),
    )
    return target_schema.model_validate_json(response.text)

def extract_with_openai(uploaded_file, api_key: str, target_schema, system_instruction: str):
    client = OpenAI(api_key=api_key)
    
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    mime_type = "application/pdf" if ext == ".pdf" else f"image/{ext.replace('.', '')}"
    
    file_bytes = uploaded_file.getvalue()
    base64_image = base64.b64encode(file_bytes).decode('utf-8')
    
    # --- TYPO FIXED HERE ON LINE 107 ---
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_instruction},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ],
            }
        ],
        response_format=target_schema,
        temperature=0.0
    )
    return response.choices[0].message.parsed

def compile_to_dataframe(extracted_data_list, mode: str) -> pd.DataFrame:
    rows = []
    
    if mode == "Purchase Invoices":
        for inv in extracted_data_list:
            for item in inv.line_items:
                rows.append({
                    "Supplier Name": inv.supplier_name,
                    "Invoice Number": inv.invoice_number,
                    "Date": inv.invoice_date,
                    "Item Name": item.item_name,
                    "Quantity": item.quantity,
                    "Rate": item.rate,
                    "Item Amount": item.item_amount,
                    "CGST Amount": item.cgst_amount,
                    "SGST Amount": item.sgst_amount,
                    "IGST Amount": item.igst_amount,
                    "Total Invoice Amount": inv.total_amount
                })
                
    elif mode == "Bank Statements":
        for statement in extracted_data_list:
            for tx in statement.transactions:
                rows.append({
                    "Bank Name": statement.bank_name,
                    "Account Holder": statement.account_holder_name,
                    "Date": tx.transaction_date,
                    "Particulars / Description": tx.particulars,
                    "Debit Amount (Withdrawal)": tx.debit_amount,
                    "Credit Amount (Deposit)": tx.credit_amount,
                    "Closing Balance": tx.closing_balance
                })
                
    elif mode == "Sale Invoices":
        for sale in extracted_data_list:
            for item in sale.line_items:
                rows.append({
                    "Invoice Date": sale.invoice_date,
                    "Invoice Number": sale.invoice_number,
                    "Buyer Name": sale.buyer_name,
                    "Buyer GST Number": sale.buyer_gst_number,
                    "Item Name": item.item_name,
                    "Quantity": item.quantity,
                    "Rate": item.rate,
                    "Item Amount": item.item_amount,
                    "IGST Amount": item.igst_amount,
                    "CGST Amount": item.cgst_amount,
                    "SGST Amount": item.sgst_amount,
                    "Total Amount": sale.total_amount
                })
                
    return pd.DataFrame(rows)

# =====================================================================
# 3. STREAMLIT USER INTERFACE DESIGN
# =====================================================================
st.set_page_config(page_title="Universal AI Data Extractor", page_icon="📊", layout="wide")
st.title("📊 Multi-Format AI Intelligent Data Processing Platform")

if "platform_choice" not in st.session_state:
    st.session_state["platform_choice"] = None
if "saved_key" not in st.session_state:
    st.session_state["saved_key"] = ""

# --- LOGIN CONTROL DESK ---
if not st.session_state["saved_key"]:
    st.subheader("⚙️ Step 1: Select AI Infrastructure Platform")
    selected_platform = st.selectbox(
        "Choose AI Engine Vendor Core Architecture:",
        ["-- Select Platform --", "Google Gemini AI Cloud", "OpenAI Platform"]
    )
    
    if selected_platform != "-- Select Platform --":
        st.markdown(f"### Authorization Credentials for **{selected_platform}**")
        input_key = st.text_input("Paste Secret API Key String Token:", type="password")
        
        if st.button("Unlock Multi-Document Extraction Engine"):
            if input_key.strip():
                st.session_state["platform_choice"] = selected_platform
                st.session_state["saved_key"] = input_key.strip()
                st.success("Platform initialized successfully!")
                st.rerun()
            else:
                st.error("Invalid token input layout.")

# --- CORE PROCESSING INTERFACE ---
else:
    with st.sidebar:
        st.success(f"🔒 AI Engine Status: ACTIVE")
        st.info(f"Vendor: {st.session_state['platform_choice']}")
        st.markdown("---")
        st.subheader("📁 Processing Profile Configuration")
        
        doc_mode = st.selectbox(
            "Select Document Data Source Profile:",
            ["Purchase Invoices", "Bank Statements", "Sale Invoices"]
        )
        
        st.markdown("---")
        if st.button("🔄 Disconnect / Swap AI Engine Configuration"):
            st.session_state["platform_choice"] = None
            st.session_state["saved_key"] = ""
            st.rerun()

    st.markdown(f"### Active Mode: **{doc_mode} Extraction Execution Workspace**")
    st.write(f"Drop your files below. The platform will automatically extract, format, and structure your data to match your requirements.")
    
    uploaded_files = st.file_uploader(
        f"Drag and drop your {doc_mode} documents here:", 
        type=["pdf", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"📂 Cached {len(uploaded_files)} source files inside extraction stream.")
        
        if st.button(f"🚀 Execute Batch {doc_mode} Extraction", type="primary"):
            all_parsed_data = []
            processed_file_hashes = {}  # Local cache tracking fingerprints
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if doc_mode == "Purchase Invoices":
                chosen_schema = PurchaseInvoiceData
                prompt_instruction = "Extract all matching lines seamlessly. Identify the vendor/seller as supplier_name. Mark unrecorded tax values as 0.0."
            elif doc_mode == "Bank Statements":
                chosen_schema = BankStatementData
                prompt_instruction = "Extract every single transaction item sequentially from the statement ledger. Do not skip any rows. Parse data precisely into the schema formats."
            elif doc_mode == "Sale Invoices":
                chosen_schema = SaleInvoiceData
                prompt_instruction = "Extract details from this sale bill/invoice. Look for the Customer or Buyer name and assign it to buyer_name. Identify the buyer's GSTIN/GST number for buyer_gst_number. Extract item grids carefully. Mark absent tax values as 0.0."

            for index, file in enumerate(uploaded_files):
                file_bytes = file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()
                
                if file_hash in processed_file_hashes:
                    status_text.text(f"🛑 Skipped Duplicate File: {file.name}")
                    st.warning(f"⚠️ **{file.name}** is a duplicate file. Re-used previous extraction data to save tokens.")
                    all_parsed_data.append(processed_file_hashes[file_hash])
                    progress_bar.progress((index + 1) / len(uploaded_files))
                    continue
                
                status_text.text(f"AI parsing document ({index+1}/{len(uploaded_files)}): {file.name}...")
                try:
                    if "Gemini" in st.session_state["platform_choice"]:
                        result = extract_with_gemini(file, st.session_state["saved_key"], chosen_schema, prompt_instruction)
                    else:
                        result = extract_with_openai(file, st.session_state["saved_key"], chosen_schema, prompt_instruction)
                    
                    all_parsed_data.append(result)
                    processed_file_hashes[file_hash] = result
                    
                except Exception as e:
                    st.error(f"Error handling processing pipeline on '{file.name}': {e}")
                
                progress_bar.progress((index + 1) / len(uploaded_files))
                
            status_text.text("✨ Conversions complete! Structuring master data reports...")
            
            if all_parsed_data:
                final_df = compile_to_dataframe(all_parsed_data, doc_mode)
                
                if doc_mode in ["Purchase Invoices", "Sale Invoices"] and not final_df.empty:
                    final_df.drop_duplicates(inplace=True)

                st.success(f"🎉 Integrated {doc_mode} Ledger Master Report Generated Successfully!")
                st.subheader("📋 Consolidated Live Preview Window")
                st.dataframe(final_df, use_container_width=True)
                
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    final_df.to_excel(writer, index=False)
                
                st.download_button(
                    label=f"📥 Download Consolidated {doc_mode} Report (.xlsx)",
                    data=buffer.getvalue(),
                    file_name=f"consolidated_{doc_mode.lower().replace(' ', '_')}_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
