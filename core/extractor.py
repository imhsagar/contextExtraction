import os
import sys
import logging
import base64
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

import numpy as np
import pdfplumber
import camelot
from pdf2image import convert_from_path
import pytesseract
import cv2
import requests

from pipeline.schemas import TaskSchema, RuleSchema
from core.utils import parse_utils, merge_utils
from core.utils.table_parser import LLMTableParser
from core.llm_client import LLMClient

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
load_dotenv(project_root / '.env')

logger = logging.getLogger(__name__)

class DocumentExtractor:
    def __init__(self, ocr_if_needed: bool = True):
        self.ocr_if_needed = ocr_if_needed
        self.llm_parser = LLMTableParser()
        self.llm_client = LLMClient()

        # Load vision model config from .env
        self.vision_api_url = os.getenv("VISION_LLM_API_URL", "http://localhost:1234/v1/chat/completions")
        self.vision_model_name = os.getenv("VISION_MODEL_NAME", "qwen2.5-vl-7b-instruct")

    def _image_to_base64(self, image_path: str) -> str:
        """Helper to convert saved image to base64 string for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _extract_with_vision_model(self, image_path: str, prompt_text: str) -> str:
        """
        Sends an image to the local Vision-Language Model (VLM) for extraction.
        """
        base64_image = self._image_to_base64(image_path)

        payload = {
            "model": self.vision_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1  # Low temp for factual data extraction
        }

        try:
            response = requests.post(self.vision_api_url, headers={"Content-Type": "application/json"}, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"[VISION] API Call failed: {e}")
            return ""

    def extract_project_schedule_vision(self, file_path: str) -> List[TaskSchema]:
        """
        New VLM-based extractor. Replaces the old 'hybrid' text parser.
        Converts PDF -> Images -> Qwen-VL -> JSON.
        """
        logger.info(f"[VISION] Starting extraction for {file_path}")

        # 1. Convert PDF Pages to Images
        try:
            images = convert_from_path(file_path, dpi=300)
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []

        all_tasks = []
        temp_dir = project_root / "temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        # 2. Process each page as an image
        for i, img in enumerate(images):
            temp_img_path = temp_dir / f"page_{i}.jpg"
            img.save(temp_img_path, "JPEG")

            logger.info(f"[VISION] Processing Page {i+1}...")

            # Prompt optimized for Vision Models reading Gantt/Tables
            prompt = """
            Analyze this Project Schedule document image.
            Extract all rows from the table.
            
            Return the data strictly as a JSON object with this key: "tasks".
            Each task must have:
            - "task_id": (string/number from ID column)
            - "task_name": (text from Task Name column)
            - "duration_days": (integer, remove 'days' text)
            - "start_date": (YYYY-MM-DD format if possible, else raw string)
            - "finish_date": (YYYY-MM-DD format if possible, else raw string)

            If a row is a summary or header, ignore it.
            Do not include markdown formatting (```json), just the raw JSON.
            """

            response_text = self._extract_with_vision_model(str(temp_img_path), prompt)

            # 3. Parse JSON Response
            try:
                # Clean markdown code blocks if the model adds them
                clean_json = response_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)

                for item in data.get("tasks", []):
                    # Validate against Schema
                    try:
                        task = TaskSchema(
                            task_id=str(item.get("task_id", "")),
                            task_name=item.get("task_name"),
                            duration_days=item.get("duration_days"),
                            start_date=item.get("start_date"),
                            finish_date=item.get("finish_date")
                        )
                        all_tasks.append(task)
                    except Exception as ve:
                        logger.warning(f"Skipping invalid task: {ve}")

            except json.JSONDecodeError:
                logger.error(f"[VISION] Failed to parse JSON from Page {i+1}. Raw: {response_text[:100]}...")

            # Cleanup temp image
            os.remove(temp_img_path)

        logger.info(f"[VISION] Completed. Extracted {len(all_tasks)} valid tasks.")
        return all_tasks

    def extract_ura_rules_vision(self, file_path: str) -> List[RuleSchema]:
        """
        Extracts rules from URA circulars using Vision to capture text + diagrams context.
        """
        logger.info(f"[URA-VISION] Starting extraction for {file_path}")

        images = convert_from_path(file_path, dpi=300)
        temp_dir = project_root / "temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        all_rules = []

        for i, img in enumerate(images):
            temp_img_path = temp_dir / f"ura_page_{i}.jpg"
            img.save(temp_img_path, "JPEG")

            prompt = """
            This is a regulatory document about Gross Floor Area (GFA) definitions.
            Extract 'Regulatory Rules' found on this page.
            Pay attention to text descriptions AND diagrams (e.g. "GFA is measured to the middle of the wall").
            
            Return strictly JSON: { "rules": [ { "rule_id": "...", "rule_summary": "...", "measurement_basis": "..." } ] }
            If no rules are found, return { "rules": [] }.
            """

            response_text = self._extract_with_vision_model(str(temp_img_path), prompt)

            try:
                clean_json = response_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)

                for item in data.get("rules", []):
                    try:
                        rule = RuleSchema(
                            rule_id=str(item.get("rule_id", "General")),
                            rule_summary=str(item.get("rule_summary", "")),
                            measurement_basis=str(item.get("measurement_basis", "N/A"))
                        )
                        all_rules.append(rule)
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"[URA-VISION] Failed page {i}: {e}")

            os.remove(temp_img_path)

        return all_rules

    # Keep the old image extraction for debug/reference if needed
    def extract_images_from_ura(self, file_path: str, output_dir: str = "extracted_images"):
        # Create output directory relative to project root
        out_path = project_root / output_dir
        if not out_path.exists():
            os.makedirs(out_path)

        logger.info(f"Processing images for: {file_path}")
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    im = page.to_image(resolution=300)

                    # Construct output path
                    image_filename = out_path / f"ura_page_{i+1}.png"
                    im.save(str(image_filename))
                    logger.info(f"Saved image: {image_filename}")
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent

    # TEST SCHEDULE (Vision)
    schedule_path = base / "data" / "Project schedule document.pdf"
    if schedule_path.exists():
        ext = DocumentExtractor()
        # Use the new VISION method
        res = ext.extract_project_schedule_vision(str(schedule_path))
        print(f"\n--- [Schedule Vision] Extracted {len(res)} tasks ---")
        if res: print(f"Sample: {res[0].dict()}")

    # TEST URA (Vision)
    ura_path = base / "data" / "URA-Circular on GFA area definition.pdf"
    if ura_path.exists():
        ext = DocumentExtractor()
        # Use the new VISION method
        rules = ext.extract_ura_rules_vision(str(ura_path))
        print(f"\n--- [URA Vision] Extracted {len(rules)} rules ---")
        if rules: print(f"Sample: {rules[0].dict()}")