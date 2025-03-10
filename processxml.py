#python processxml.py --log-level INFO

import argparse
import glob
import json
import logging
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocessing import Pool, cpu_count
import csv

# Set up constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(BASE_DIR, "A_Test_Collection_for_Matching_Patient_to_Clinical_Trials", "data", "clinicaltrials.gov-16_dec_2015", "clinicaltrials.gov-16_dec_2015")
OUTPUT = os.path.join(BASE_DIR, "TrialGPT", "dataset", "mine")
OUTPUT_FILE = os.path.join(OUTPUT, "corpus.jsonl")

# Set up logging
def setup_logging(args):
    log_file = os.path.join(BASE_DIR, f"processxml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    console_level = args.log_level  # This will be WARNING by default

    # Set up file handler (always DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Set up console handler (level based on user input)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # This ensures all messages are processed
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Console logging level set to: {logging.getLevelName(console_level)}")
    logging.debug("File logging level set to: DEBUG")

def debug_timer(func):
    def wrapper(*args, **kwargs):
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Process clinical trial XML files.")
    parser.add_argument('--log-level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: WARNING)')
    return parser.parse_args()


def clean_text(text):
    if text is None:
        return ""
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Replace Unicode characters with ASCII equivalents
    text = text.replace('\u2264', '<=').replace('\u2265', '>=')
    return text


# @debug_timer
def process_xml_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        nct_id = root.find('id_info/nct_id').text
        title = clean_text(root.find('brief_title').text)

        brief_summary = root.find('brief_summary/textblock')
        brief_summary_text = clean_text(brief_summary.text) if brief_summary is not None else ""

        detailed_description = root.find('detailed_description/textblock')
        detailed_description_text = clean_text(detailed_description.text) if detailed_description is not None else ""

        criteria = root.find('.//criteria/textblock')
        criteria_text = clean_text(criteria.text) if criteria is not None else ""

        # Split criteria text into inclusion and exclusion
        inclusion_text = ""
        exclusion_text = ""
        if "Inclusion Criteria:" in criteria_text and "Exclusion Criteria:" in criteria_text:
            parts = criteria_text.split("Exclusion Criteria:")
            inclusion_text = parts[0].split("Inclusion Criteria:")[-1].strip()
            exclusion_text = parts[1].strip()
        elif "Inclusion Criteria:" in criteria_text:
            inclusion_text = criteria_text.split("Inclusion Criteria:")[-1].strip()
        elif "Exclusion Criteria:" in criteria_text:
            exclusion_text = criteria_text.split("Exclusion Criteria:")[-1].strip()

        # Clean up inclusion and exclusion criteria
        inclusion_text = clean_text(inclusion_text)
        exclusion_text = clean_text(exclusion_text)

        enrollment = root.find('enrollment')
        enrollment_value = enrollment.text if enrollment is not None else "0"

        intervention_elements = root.findall('.//intervention/intervention_name')
        drugs_list = [clean_text(elem.text) for elem in intervention_elements if elem.text]

        condition_elements = root.findall('condition')
        diseases_list = [clean_text(elem.text) for elem in condition_elements if elem.text]

        phase_element = root.find('phase')
        phase = clean_text(phase_element.text) if phase_element is not None else ""

        data = {
            "_id": nct_id,
            "title": title,
            #"text": f"Summary: {brief_summary_text}\nInclusion criteria: {inclusion_text}\nExclusion criteria: {exclusion_text}",
            "metadata": {
                #"brief_title": title,
                "phase": phase,
                "drugs": str(drugs_list),
                "drugs_list": drugs_list,
                # "diseases": str(diseases_list),
                "diseases_list": diseases_list,
                "enrollment": enrollment_value,
                "inclusion_criteria": inclusion_text,
                "exclusion_criteria": exclusion_text,
                "brief_summary": brief_summary_text,
                "detailed_description": detailed_description_text
            }
        }

        return data

    except ET.ParseError:
        logging.error(f"Invalid XML file: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return None


def copy_and_transform_files():
    source_dir = os.path.join(BASE_DIR, "A_Test_Collection_for_Matching_Patient_to_Clinical_Trials", "data")
    files_to_copy = [
        ("adhoc-queries.json", "queries.jsonl"),
        ("qrels-clinical_trials.txt", "test.tsv")
    ]

    for source_file, dest_file in files_to_copy:
        source_path = os.path.join(source_dir, source_file)
        destination_path = os.path.join(OUTPUT, dest_file)

        try:
            if source_file == "adhoc-queries.json":
                # Read the original JSON file
                with open(source_path, 'r', encoding='utf-8') as f:
                    queries = json.load(f)

                # Transform the queries and write to JSONL format
                with open(destination_path, 'w', encoding='utf-8') as f:
                    for query in queries:
                        transformed_query = {
                            "_id": f"sigir-{query['qId'][4:].replace('-', '')}",
                            "text": query['description']
                        }
                        json.dump(transformed_query, f, ensure_ascii=False)
                        f.write('\n')

                logging.info(f"Successfully transformed {source_file} and saved as {dest_file} in {OUTPUT}")

            elif source_file == "qrels-clinical_trials.txt":
                # Read the original file and transform
                with open(source_path, 'r', encoding='utf-8') as infile, \
                        open(destination_path, 'w', encoding='utf-8', newline='') as outfile:

                    tsv_writer = csv.writer(outfile, delimiter='\t')

                    # Write the header
                    tsv_writer.writerow(['query-id', 'corpus-id', 'score'])

                    # Process and write the data
                    for line in infile:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            query_id = f"sigir-{parts[0]}"
                            corpus_id = parts[2]
                            score = parts[3]
                            tsv_writer.writerow([query_id, corpus_id, score])

                logging.info(f"Successfully transformed {source_file} and saved as {dest_file} in {OUTPUT}")

            else:
                # For any other files, just copy without modification
                shutil.copy2(source_path, destination_path)
                logging.info(f"Successfully copied {source_file} to {OUTPUT} as {dest_file}")

        except FileNotFoundError:
            logging.error(f"Source file not found: {source_path}")
        except PermissionError:
            logging.error(f"Permission denied when copying {source_file}")
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON in {source_file}")
        except Exception as e:
            logging.error(f"Error processing {source_file}: {str(e)}")

# @debug_timer
def main():
    args = parse_args()
    setup_logging(args)

    logging.debug(f"BASE_DIR: {BASE_DIR}")

    shutil.rmtree(OUTPUT, ignore_errors=True)
    os.makedirs(OUTPUT, exist_ok=True)
    logging.info(f"Removed and recreated directory: {OUTPUT}")

    xml_files = glob.glob(os.path.join(RUN_DIR, "*.xml"))
    logging.info(f"Found {len(xml_files)} XML files to process")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_xml_file, xml_files)

    valid_results = [result for result in results if result is not None]

    with open(OUTPUT_FILE, 'w') as f:
        for result in valid_results:
            json.dump(result, f)
            f.write('\n')

    logging.info(f"Processed {len(valid_results)} files successfully. Output saved to {OUTPUT_FILE}")

    # Copy additional files after processing XML files
    copy_and_transform_files()

if __name__ == "__main__":
    main()