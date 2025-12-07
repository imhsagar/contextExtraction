# core/extractor.py
import os
import sys
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv

import numpy as np
import pdfplumber
import camelot
from pdf2image import convert_from_path
import pytesseract
import cv2
from pipeline.schemas import TaskSchema, RuleSchema
from core.utils import parse_utils, merge_utils
from core.utils.table_parser import LLMTableParser
from core.llm_client import LLMClient  # Needed for URA text extraction

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
load_dotenv(project_root / '.env')

logger = logging.getLogger(__name__)

class DocumentExtractor:
    def __init__(self, ocr_if_needed: bool = True):
        self.ocr_if_needed = ocr_if_needed
        # Initialize Parsers
        self.llm_parser = LLMTableParser()
        # Direct client for unstructured text (URA rules)
        self.llm_client = LLMClient()

    def extract_project_schedule_hybrid(self, file_path: str) -> List[TaskSchema]:
        logger.info(f"[HYBRID] Starting extraction for {file_path}")

        # 1. Collect Raw Tables (PDF Structure)
        raw_tables = self._collect_raw_tables(file_path)
        logger.info(f"[HYBRID] Found {len(raw_tables)} raw tables")

        # 2. Rule-Based Extraction
        rule_tasks_objs = []
        for idx, table in enumerate(raw_tables):
            try:
                tasks = parse_utils._parse_raw_table_rows(table, file_path, page_num=idx+1)
                rule_tasks_objs.extend(tasks)
            except Exception as e:
                logger.warning(f"Rule-based parse failed for table {idx}: {e}")

        # 3. LLM Extraction
        llm_tasks_dicts = []
        for idx, table in enumerate(raw_tables):
            try:
                tasks = self.llm_parser.parse_table_hybrid(table, page_num=idx+1)
                llm_tasks_dicts.extend(tasks)
            except Exception as e:
                logger.error(f"LLM parse failed for table {idx}: {e}")

        # 4. Merge Logic
        logger.info(f"[HYBRID] Merging {len(rule_tasks_objs)} rule tasks and {len(llm_tasks_dicts)} LLM tasks")
        rule_dicts = [parse_utils.safe_task_to_dict(t) for t in rule_tasks_objs]
        final_schemas = merge_utils.merge_tasks(rule_dicts + llm_tasks_dicts)

        logger.info(f"[HYBRID] Completed. Extracted {len(final_schemas)} valid tasks.")
        return final_schemas

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

    def extract_ura_rules_with_llm(self, file_path: str) -> List[RuleSchema]:

        logger.info("Extracting text for URA Rule processing...")

        text_content = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text_content += (page.extract_text() or "") + "\n"
        except Exception as e:
            logger.error(f"PDF Text extraction failed: {e}")
            return []

        chunk_size = 1500
        chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]

        logger.info(f"[URA] Text split into {len(chunks)} chunks.")
        valid_rules = []

        for i, chunk in enumerate(chunks):
            prompt = f"""
                Extract 'Regulatory Rules' from this text.
                Return a JSON object: {{ "rules": [ {{ "rule_id": "...", "rule_summary": "...", "measurement_basis": "..." }} ] }}
                
                Text:
                {chunk} 
                """

            try:
                data = self.llm_client.ask_json(prompt, temperature=0.0, max_tokens=2000)

                count = 0
                for item in data.get("rules", []):
                    try:
                        rule = RuleSchema(
                            rule_id=str(item.get("rule_id", "Unknown")),
                            rule_summary=str(item.get("rule_summary", "")),
                            measurement_basis=str(item.get("measurement_basis", "N/A"))
                        )
                        valid_rules.append(rule)
                        count += 1
                    except Exception:
                        continue
                logger.info(f"[URA] Chunk {i+1}/{len(chunks)}: Found {count} rules.")

            except Exception as e:
                logger.error(f"[URA] Chunk {i+1} failed: {e}")

        logger.info(f"[URA] Total Extracted: {len(valid_rules)} rules.")
        return valid_rules

    def _collect_raw_tables(self, file_path: str) -> List[List[List[str]]]:
        """
        Strategy: Camelot (Lattice) -> PDFPlumber -> OCR Fallback
        """
        raw_tables = []

        # A. Camelot (Best for Grid Tables)
        try:
            # Check if file is encrypted or readable first
            tables = camelot.read_pdf(file_path, flavor='lattice', pages='all')
            for t in tables:
                raw_tables.append(t.df.values.tolist())
        except Exception as e:
            logger.debug(f"Camelot lattice failed: {e}")

        # B. PDFPlumber (Fallback for stream/borderless tables)
        if not raw_tables:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_tables()
                        for t in extracted:
                            if t: raw_tables.append(t)
            except Exception as e:
                logger.error(f"PDFPlumber failed: {e}")

        # C. OCR Fallback (Last Resort)
        if not raw_tables and self.ocr_if_needed:
            logger.info("No tables found. Running OCR...")
            try:
                images = convert_from_path(file_path)
                for i, img in enumerate(images):
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray)

                    parsed = parse_utils._parse_text_block_for_tasks(text, file_path, page_num=i+1)
                    if parsed:
                        pseudo_table = []
                        for p in parsed:
                            d = parse_utils.safe_task_to_dict(p)
                            pseudo_table.append([
                                str(d.get('task_id', '')),
                                d.get('task_name', ''),
                                str(d.get('duration_days', '')),
                                str(d.get('start_date', '')),
                                str(d.get('finish_date', ''))
                            ])
                        if pseudo_table:
                            raw_tables.append(pseudo_table)
            except Exception as e:
                logger.error(f"OCR failed: {e}")

        return raw_tables

if __name__ == "__main__":
    # Test harness
    base = Path(__file__).resolve().parent.parent

    schedule_path = base / "data" / "Project schedule document.pdf"
    if schedule_path.exists():
        ext = DocumentExtractor(ocr_if_needed=True)
        res = ext.extract_project_schedule_hybrid(str(schedule_path))
        print(f"\n--- [Schedule] Extracted {len(res)} tasks ---")
        if res: print(f"Sample: {res[0].dict()}")

    ura_path = base / "data" / "URA-Circular on GFA area definition.pdf"
    if ura_path.exists():
        print(f"\n--- [URA] Processing ---")
        ext = DocumentExtractor(ocr_if_needed=False) # OCR usually not needed for text PDFs
        ext.extract_images_from_ura(str(ura_path))
        rules = ext.extract_ura_rules_with_llm(str(ura_path))
        print(f"--- [URA] Extracted {len(rules)} rules ---")
        if rules: print(f"Sample: {rules[0].dict()}")