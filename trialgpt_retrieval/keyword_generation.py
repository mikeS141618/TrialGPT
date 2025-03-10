#!/usr/bin/env python3

"""
Generate search keywords for patient descriptions using specified model and corpus.
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import setup_model, generate_response
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Also set up a file handler for JSON fix attempts
fix_logger = logging.getLogger('json_fix_logger')
fix_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('json_fix_attempts.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
fix_logger.addHandler(file_handler)


def fix_json_in_conversation(model_type, model_instance, messages, invalid_output, model_name=None, entry_id=None):
    """Ask the model to fix the JSON output within the existing conversation."""
    messages.append({"role": "assistant", "content": invalid_output})
    messages.append({"role": "user", "content": "The JSON is invalid. Please fix any errors and return only the corrected JSON:"})

    fixed_output = generate_response(model_type, model_instance, messages, model_name)

    # Log the fix attempt
    fix_logger.info(f"JSON fix attempted for entry {entry_id}")
    fix_logger.info(f"Original output: {invalid_output}")
    fix_logger.info(f"Fixed output: {fixed_output}")

    return fixed_output.strip()


def parse_json_with_conversation_fix(output, model_type, model_instance, messages, model_name=None, entry_id=None):
    """Parse JSON output with model-based error correction within the conversation."""
    try:
        # First, try parsing the output as-is
        return json.loads(output), False
    except json.JSONDecodeError:
        # If parsing fails, ask the model to fix it within the conversation
        fixed_output = fix_json_in_conversation(model_type, model_instance, messages, output, model_name, entry_id)
        try:
            # Try parsing the fixed output
            return json.loads(fixed_output), True
        except json.JSONDecodeError:
            # If it still fails, return None
            return None, True

def parse_arguments_kg():
    """
    Parse command-line arguments for the keyword generation script.

    This function sets up the argument parser and defines the required and optional
    arguments for the script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate search keywords for patient descriptions.")

    # Required arguments
    parser.add_argument("-c", "--corpus", required=True, help="The corpus to process: trec_2021, trec_2022, or sigir")
    parser.add_argument("-m", "--model", required=True, help="The model to use for generating keywords")
    parser.add_argument("-g", "--num_gpus", help="The number of GPUs to use for model distribution")
    # Optional arguments
    parser.add_argument("-d", "--checkpoint_dir", help="Checkpoint directory for Llama models")
    parser.add_argument("-q", "--quantize", action="store_true", help="Use 8-bit quantization for Llama models")

    return parser.parse_args()


def get_keyword_generation_messages(note):
    """
    Prepare the messages for keyword generation based on a patient note.

    Args:
        note (str): The patient description.

    Returns:
        list: A list of message dictionaries for the AI model.
    """
    system = """You are an AI assistant specializing in clinical trial matching. Your task is to analyze patient descriptions and extract key information that would be relevant for finding suitable clinical trials. Focus on accuracy, relevance, and comprehensive analysis in your assessment."""

    prompt = f"""Please analyze the following patient description for clinical trial matching:

    {note}

    ### Instructions:
    1. Summarize the patient's clinical presentation, including key demographic information, presenting symptoms, and relevant medical history.
    2. Enumerate all clinically relevant conditions, characteristics, and factors that could influence clinical trial eligibility or suitability. Use standardized medical terminology. Rank these by clinical significance and relevance to potential trial matching.
    3. Include any additional clinical notes that might be pertinent for trial matching but don't fit into the main conditions list.

    ### Output a JSON object in this format:
    **Provide ONLY a valid JSON object** with the following structure:
    {{
      "summary": "Concise clinical summary including key demographics, presenting symptoms, and relevant history",
      "conditions": ["Condition 1", "Condition 2", ...],
      "notes": "Additional clinically relevant information for trial matching"
    }}

    ### Important:
    - Include only clinical information explicitly stated or strongly implied in the description.
    - If there is uncertainty about a condition, include it only if it is explicitly mentioned or strongly implied, noting the uncertainty if appropriate.
    - **Do NOT include any text outside of the JSON object.** This means no notes, explanations, headers, or footers outside the JSON.

    Please process the patient description and respond with the JSON object.
    """

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]


def main(args):
    """
    Generate search keywords for patient descriptions using specified model and corpus.

    This function processes patient descriptions from a given corpus using either GPT or Llama models
    to generate relevant medical keywords. It saves the results to a JSON file.
    """
    outputs = {}
    failed_outputs = {}

    model_type, model_instance = setup_model(args.model, args.num_gpus, args.checkpoint_dir, args.quantize)

    # Count total lines in the input file for progress tracking
    with open(f"dataset/{args.corpus}/queries.jsonl", "r") as f:
        total_lines = sum(1 for _ in f)

    # Process each query in the input file
    with open(f"dataset/{args.corpus}/queries.jsonl", "r") as f:
        fix_count = 0
        for line in tqdm(f, total=total_lines, desc=f"Processing {args.corpus} queries"):
            try:
                entry = json.loads(line)
                messages = get_keyword_generation_messages(entry["text"])
                output = generate_response(model_type, model_instance, messages, args.model)

                parsed_output, was_fixed = parse_json_with_conversation_fix(output, model_type, model_instance,
                                                                            messages, args.model, entry["_id"])

                if was_fixed:
                    fix_count += 1

                if parsed_output is not None:
                    outputs[entry["_id"]] = parsed_output
                else:
                    logger.warning(f"Failed to parse JSON for entry {entry['_id']} even after model fix attempt.")
                    failed_outputs[entry["_id"]] = {
                        "error": "Failed to parse JSON after model fix attempt",
                        "raw_output": output
                    }
            except Exception as e:
                logger.error(f"Error processing entry {entry['_id']}: {str(e)}")
                failed_outputs[entry["_id"]] = {
                    "error": str(e),
                    "raw_entry": line
                }

        # After processing all entries, log the summary
        logger.info(f"Total entries processed: {len(outputs) + len(failed_outputs)}")
        logger.info(f"Successful entries: {len(outputs)}")
        logger.info(f"Failed entries: {len(failed_outputs)}")
        logger.info(f"Entries requiring JSON fix: {fix_count}")

    # Save successful outputs
    output_file = f"results/retrieval_keywords_{args.model}_{args.corpus}.json"
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)
    print(f"Results saved to {output_file}")

    # Save failed outputs
    failed_output_file = f"results/failed_retrieval_keywords_{args.model}_{args.corpus}.json"
    with open(failed_output_file, "w") as f:
        json.dump(failed_outputs, f, indent=4)
    print(f"Failed results saved to {failed_output_file}")

    # Print summary
    print(f"Total entries processed: {len(outputs) + len(failed_outputs)}")
    print(f"Successful entries: {len(outputs)}")
    print(f"Failed entries: {len(failed_outputs)}")


if __name__ == "__main__":
    args = parse_arguments_kg()
    main(args)
