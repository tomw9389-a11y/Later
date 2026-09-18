import os
import json
import time
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

load_dotenv()
client = genai.Client()

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class EmailClassification(BaseModel):
    category: str = Field(description="Must be one of: 'document_comparison', 'new_si_request', 'invoice_query', 'general_message', 'spam'")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")

class ShipmentDetails(BaseModel):
    shipper: Optional[str] = Field(description="Name of the shipper. Null if unreadable.")
    consignee: Optional[str] = Field(description="Name of the consignee. Null if unreadable.")
    notify_party: Optional[str] = Field(description="Name of the notify party. Null if unreadable.")
    port_of_loading: Optional[str] = Field(description="Port of loading. Null if unreadable.")
    port_of_discharge: Optional[str] = Field(description="Port of discharge. Null if unreadable.")
    container_count: Optional[int] = Field(description="Total number of containers. Null if unreadable.")
    gross_weight_kg: Optional[float] = Field(description="Total gross weight strictly in kilograms. Null if unreadable.")
    requires_human_review: bool = Field(description="Set to true if document is blurry, contradictory, or missing data.")

# ---------------------------------------------------------
# Core System Capabilities
# ---------------------------------------------------------
class DocumentVerificationSystem:
    def __init__(self, data_dir: str):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.inbox_dir = os.path.join(base_path, data_dir, "inbox")
        self.attachments_dir = os.path.join(base_path, data_dir, "attachments")

    def classify_email(self, email_data: Dict) -> EmailClassification:
        prompt = f"Analyze the following email and classify it into one of these exact categories: 'document_comparison', 'new_si_request', 'invoice_query', 'general_message', 'spam'.\nSubject: {email_data.get('subject', '')}\nBody: {email_data.get('body', '')}"
        
        # Retry logic for 503 errors
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=EmailClassification,
                    ),
                )
                return EmailClassification.model_validate_json(response.text)
            except Exception as e:
                logging.warning(f"Classification API Error. Retrying in 5s... (Attempt {attempt+1}/3)")
                time.sleep(5)
                
        # Fallback if all retries fail
        return EmailClassification(category="general_message", confidence=0.0)

    def extract_document_data(self, filename: str) -> ShipmentDetails:
        file_path = os.path.join(self.attachments_dir, filename)
        if not os.path.exists(file_path):
            logging.warning(f"Attachment missing: {filename}")
            return ShipmentDetails(requires_human_review=True)

        for attempt in range(3):
            try:
                uploaded_file = client.files.upload(file=file_path)
                prompt = "Extract the following shipping details from this document: shipper, consignee, notify party, port of loading, port of discharge, container count, and gross weight in kilograms. If a value is obscured or missing, leave it null. If the document is unreadable, set requires_human_review to true."
                
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ShipmentDetails,
                    ),
                )
                client.files.delete(name=uploaded_file.name)
                return ShipmentDetails.model_validate_json(response.text)
            except Exception as e:
                logging.warning(f"Extraction API Error on {filename}. Retrying in 5s... (Attempt {attempt+1}/3)")
                time.sleep(5)

        return ShipmentDetails(requires_human_review=True)

    def normalize_string(self, val: Any) -> str:
        if val is None:
            return ""
        return str(val).lower().strip().replace('\n', ' ')

    def compare_documents(self, si_data: ShipmentDetails, bl_data: ShipmentDetails) -> Dict[str, Any]:
        mismatches = {}
        fields_to_check = ['shipper', 'consignee', 'notify_party', 'port_of_loading', 'port_of_discharge', 'container_count', 'gross_weight_kg']
        for field in fields_to_check:
            si_val = getattr(si_data, field)
            bl_val = getattr(bl_data, field)
            if isinstance(si_val, str) and isinstance(bl_val, str):
                if self.normalize_string(si_val) != self.normalize_string(bl_val):
                    mismatches[field] = {"SI": si_val, "BL": bl_val}
            else:
                if si_val != bl_val:
                    mismatches[field] = {"SI": si_val, "BL": bl_val}
        return mismatches

    def process_inbox(self) -> Dict[str, Any]:
        submission_result = {}
        if not os.path.exists(self.inbox_dir):
            logging.error(f"Inbox directory not found at: {self.inbox_dir}")
            return submission_result

        email_files = [f for f in os.listdir(self.inbox_dir) if f.endswith(".json")]
        
        for index, email_file in enumerate(email_files):
            email_id = email_file.replace(".json", "")
            with open(os.path.join(self.inbox_dir, email_file), 'r') as f:
                email_data = json.load(f)

            classification = self.classify_email(email_data)
            time.sleep(4) # PAUSE: Rate limiting for free tier
            
            record = {
                "category": classification.category,
                "needs_human_review": False,
                "mismatch_found": False,
                "discrepancies": {}
            }

            if classification.category == "document_comparison":
                attachments = email_data.get("attachments", [])
                si_file = next((f for f in attachments if "SI" in f), None)
                bl_file = next((f for f in attachments if "BL" in f), None)

                if not si_file or not bl_file:
                    record["needs_human_review"] = True
                else:
                    si_data = self.extract_document_data(si_file)
                    time.sleep(4) # PAUSE: Rate limiting for free tier
                    
                    bl_data = self.extract_document_data(bl_file)
                    time.sleep(4) # PAUSE: Rate limiting for free tier

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
            logging.info(f"Processed {index + 1}/{len(email_files)}: {email_id} -> {classification.category}")

        return submission_result

if __name__ == "__main__":
    system = DocumentVerificationSystem(data_dir="data")
    final_report = system.process_inbox()
    
    output_file = "sample_submission.json"
    with open(output_file, "w") as f:
        json.dump(final_report, f, indent=4)
    
    logging.info(f"Processing complete. Results written to {output_file}")