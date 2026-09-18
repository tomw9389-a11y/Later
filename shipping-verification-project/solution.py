import os
import json
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize the Gemini API (Ensure your GOOGLE_API_KEY environment variable is set)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-pro')

# ---------------------------------------------------------
# Pydantic Schemas for Structured LLM Extraction
# ---------------------------------------------------------
class EmailClassification(BaseModel):
    category: str = Field(
        description="Must be one of: 'document_comparison', 'new_si_request', 'invoice_query', 'general_message', 'spam'"
    )
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")

class ShipmentDetails(BaseModel):
    shipper: Optional[str] = Field(description="Name of the shipper. Null if unreadable.")
    consignee: Optional[str] = Field(description="Name of the consignee. Null if unreadable.")
    notify_party: Optional[str] = Field(description="Name of the notify party. Null if unreadable.")
    port_of_loading: Optional[str] = Field(description="Port of loading. Null if unreadable.")
    port_of_discharge: Optional[str] = Field(description="Port of discharge. Null if unreadable.")
    container_count: Optional[int] = Field(description="Total number of containers. Null if unreadable.")
    gross_weight_kg: Optional[float] = Field(description="Total gross weight strictly in kilograms. Null if unreadable.")
    requires_human_review: bool = Field(
        description="Set to true if the document is too blurry, contradictory, or missing critical data."
    )

# ---------------------------------------------------------
# Core System Capabilities
# ---------------------------------------------------------
class DocumentVerificationSystem:
    def __init__(self, data_dir: str):
        self.inbox_dir = os.path.join(data_dir, "inbox")
        self.attachments_dir = os.path.join(data_dir, "attachments")

    def classify_email(self, email_data: Dict) -> EmailClassification:
        """Classifies the email into one of the 5 allowed categories."""
        prompt = f"""
        Analyze the following email and classify it into one of these exact categories: 
        'document_comparison', 'new_si_request', 'invoice_query', 'general_message', 'spam'.
        
        Subject: {email_data.get('subject', '')}
        Body: {email_data.get('body', '')}
        """
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=EmailClassification
            )
        )
        return EmailClassification.model_validate_json(response.text)

    def extract_document_data(self, filename: str) -> ShipmentDetails:
        """Uploads the file (TXT, PDF, etc.) to Gemini and extracts the 7 critical fields."""
        file_path = os.path.join(self.attachments_dir, filename)
        
        if not os.path.exists(file_path):
            logging.warning(f"Attachment missing: {filename}")
            return ShipmentDetails(requires_human_review=True)

        try:
            # Upload file to Gemini API (handles PDF, TXT, DOCX visually/textually)
            uploaded_file = genai.upload_file(file_path)
            
            prompt = """
            Extract the following shipping details from this document: shipper, consignee, notify party, 
            port of loading, port of discharge, container count, and gross weight in kilograms.
            If a value is obscured or missing, leave it null. If the document is unreadable, 
            set requires_human_review to true.
            """
            
            response = model.generate_content(
                [uploaded_file, prompt],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ShipmentDetails
                )
            )
            
            # Clean up the file from Google's servers
            genai.delete_file(uploaded_file.name)
            
            return ShipmentDetails.model_validate_json(response.text)
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            return ShipmentDetails(requires_human_review=True)

    def normalize_string(self, val: Any) -> str:
        """Normalizes strings for strict comparison to reduce false discrepancies."""
        if val is None:
            return ""
        return str(val).lower().strip().replace('\n', ' ')

    def compare_documents(self, si_data: ShipmentDetails, bl_data: ShipmentDetails) -> Dict[str, Any]:
        """Compares the SI and BL fields and formats the mismatch report."""
        mismatches = {}
        fields_to_check = [
            'shipper', 'consignee', 'notify_party', 
            'port_of_loading', 'port_of_discharge', 
            'container_count', 'gross_weight_kg'
        ]

        for field in fields_to_check:
            si_val = getattr(si_data, field)
            bl_val = getattr(bl_data, field)
            
            # Standardize for comparison
            if isinstance(si_val, str) and isinstance(bl_val, str):
                if self.normalize_string(si_val) != self.normalize_string(bl_val):
                    mismatches[field] = {"SI": si_val, "BL": bl_val}
            else:
                # Numeric comparison (container count, weight)
                if si_val != bl_val:
                    mismatches[field] = {"SI": si_val, "BL": bl_val}

        return mismatches

    def process_inbox(self) -> Dict[str, Any]:
        """Main pipeline: iterates through the inbox and builds the final submission."""
        submission_result = {}

        if not os.path.exists(self.inbox_dir):
            logging.error("Inbox directory not found.")
            return submission_result

        for email_file in os.listdir(self.inbox_dir):
            if not email_file.endswith(".json"):
                continue

            email_id = email_file.replace(".json", "")
            
            with open(os.path.join(self.inbox_dir, email_file), 'r') as f:
                email_data = json.load(f)

            # Step 1: Classify
            classification = self.classify_email(email_data)
            
            record = {
                "category": classification.category,
                "needs_human_review": False,
                "mismatch_found": False,
                "discrepancies": {}
            }

            # Step 2 & 3: Extract and Compare (Only for document comparisons)
            if classification.category == "document_comparison":
                attachments = email_data.get("attachments", [])
                
                # Identify which attachment is SI and which is BL based on filenames
                si_file = next((f for f in attachments if "SI" in f), None)
                bl_file = next((f for f in attachments if "BL" in f), None)

                if not si_file or not bl_file:
                    record["needs_human_review"] = True
                else:
                    si_data = self.extract_document_data(si_file)
                    bl_data = self.extract_document_data(bl_file)

                    if si_data.requires_human_review or bl_data.requires_human_review:
                        record["needs_human_review"] = True
                    else:
                        mismatches = self.compare_documents(si_data, bl_data)
                        if mismatches:
                            record["mismatch_found"] = True
                            record["discrepancies"] = mismatches
                        else:
                            record["mismatch_found"] = False
                            record["discrepancies"] = "No mismatch detected."

            submission_result[email_id] = record
            logging.info(f"Processed {email_id} -> Category: {classification.category}")

        return submission_result

# ---------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ensure you set the directory containing 'inbox' and 'attachments' folders
    system = DocumentVerificationSystem(data_dir="data_v2")
    
    # Process the dataset
    final_report = system.process_inbox()
    
    # Output to submission file
    output_file = "sample_submission.json"
    with open(output_file, "w") as f:
        json.dump(final_report, f, indent=4)
    
    logging.info(f"Processing complete. Results written to {output_file}")