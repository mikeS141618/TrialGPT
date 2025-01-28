__author__ = "qiao"

"""
Rank the trials given the matching and aggregation results
"""

import argparse
import json
import math
import os
import sys
import traceback

from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import load_corpus_details

eps = 1e-9

def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Rank trials based on matching and aggregation results.")
    parser.add_argument("matching_results_path", help="Path to the matching results JSON file")
    parser.add_argument("agg_results_path", help="Path to the aggregation results JSON file")
    parser.add_argument("overwrite", help="Overwrite existing results (true/false)")
    parser.add_argument("corpus", help="Corpus name")
    parser.add_argument("model", help="Model name")
    return parser.parse_args()

def get_matching_score(matching):
    """
    Calculate the matching score based on inclusion and exclusion criteria.

    Args:
        matching (dict): Dictionary containing matching results for inclusion and exclusion criteria.

    Returns:
        float: The calculated matching score.
    """

    # count only the valid ones
    included = 0
    not_inc = 0
    na_inc = 0
    no_info_inc = 0

    excluded = 0
    not_exc = 0
    na_exc = 0
    no_info_exc = 0

    # first count inclusions
    for criteria, info in matching["inclusion"].items():

        if len(info) != 3:
            continue

        if info[2] == "included":
            included += 1
        elif info[2] == "not included":
            not_inc += 1
        elif info[2] == "not applicable":
            na_inc += 1
        elif info[2] == "not enough information":
            no_info_inc += 1

    # then count exclusions
    for criteria, info in matching["exclusion"].items():

        if len(info) != 3:
            continue

        if info[2] == "excluded":
            excluded += 1
        elif info[2] == "not excluded":
            not_exc += 1
        elif info[2] == "not applicable":
            na_exc += 1
        elif info[2] == "not enough information":
            no_info_exc += 1

    # get the matching score
    score = 0

    score += included / (included + not_inc + no_info_inc + eps)

    if not_inc > 0:
        score -= 1

    if excluded > 0:
        score -= 1

    return score


def get_agg_score(assessment):
    """
    Calculate the aggregation score based on relevance and eligibility scores.

    Args:
        assessment (dict): Dictionary containing relevance and eligibility scores.

    Returns:
        float: The calculated aggregation score.
    """
    try:
        rel_score = float(assessment["relevance_score_R"])
        eli_score = float(assessment["eligibility_score_E"])
    except:
        rel_score = 0
        eli_score = 0

    score = (rel_score + eli_score) / 100

    return score


def main(args):
    """
    Main function to rank trials based on matching and aggregation results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Loading the results
    with open(args.matching_results_path, 'r') as f:
        matching_results = json.load(f)
    with open(args.agg_results_path, 'r') as f:
        agg_results = json.load(f)

    # Load corpus details using the common utility function
    corpus_details = load_corpus_details(f"dataset/{args.corpus}/corpus.jsonl")

    # Load patient summaries
    with open(f"results/retrieval_keywords_{args.model}_{args.corpus}.json", 'r') as f:
        patient_summaries = json.load(f)

    # Set output directory
    output_dir = "results/trial_rankings"
    os.makedirs(output_dir, exist_ok=True)

    # Check if all_rankings.json exists and if we should overwrite
    all_rankings_file = os.path.join(output_dir, "all_rankings.json")
    if os.path.exists(all_rankings_file) and args.overwrite.lower() != 'true':
        print(f"Loading existing rankings from {all_rankings_file}")
        with open(all_rankings_file, 'r') as f:
            all_rankings = json.load(f)
    else:
        all_rankings = {}

    # For qrels-like output
    qrels_output = []

    # Loop over the patients
    for patient_id, label2trial2results in tqdm(matching_results.items(), desc="Processing patients"):
        # Skip if patient already processed and not overwriting
        if patient_id in all_rankings and args.overwrite.lower() != 'true':
            continue

        patient_summary = patient_summaries.get(patient_id, {}).get('summary', 'No summary available')
        trial2scores = {}
        try:
            for _, trial2results in label2trial2results.items():
                for trial_id, results in trial2results.items():
                    matching_score = get_matching_score(results)

                    if patient_id not in agg_results or trial_id not in agg_results[patient_id]:
                        print(f"Patient {patient_id} Trial {trial_id} not in the aggregation results.")
                        agg_score = 0
                    else:
                        agg_score = get_agg_score(agg_results[patient_id][trial_id])

                    trial_score = matching_score + agg_score
                    trial2scores[trial_id] = {
                        "matching_score": matching_score,
                        "agg_score": agg_score,
                        "trial_score": trial_score,
                        "brief_summary": corpus_details.get(trial_id, {}).get("brief_summary", "No summary available")
                    }

                    # Generate qrels-like output using the combined score
                    qrels_score = min(2, max(0, math.floor(trial_score)))
                    qrels_output.append(f"{patient_id}\t{trial_id}\t{qrels_score}")

            sorted_trial2scores = sorted(trial2scores.items(), key=lambda x: -x[1]["trial_score"])

            # Save to individual text file
            output_file = os.path.join(output_dir, f"trialranking_{patient_id}.txt")
            with open(output_file, 'w') as f:
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Patient Summary: {patient_summary}\n\n")
                f.write("Clinical trial ranking:\n")
                for trial, scores in sorted_trial2scores:
                    f.write(f"{trial}: matching_score={scores['matching_score']:.4f}, "
                            f"agg_score={scores['agg_score']:.4f}, "
                            f"trial_score={scores['trial_score']:.4f}\n")
                    f.write(f"Brief Summary: {scores['brief_summary']}\n\n")

            print(f"Ranking saved to {output_file}")

            # Add to all_rankings dictionary
            all_rankings[patient_id] = {
                "patient_summary": patient_summary,
                "trials": dict(sorted_trial2scores)
            }

        except Exception as e:
            print(f"An error occurred while processing patient {patient_id}:")
            print(traceback.format_exc())
            print("Continuing with the next patient...")

    # Save all rankings to a single JSON file
    with open(all_rankings_file, 'w') as f:
        json.dump(all_rankings, f, indent=2)

    print(f"All rankings saved to {all_rankings_file}")

    # Save qrels-like output
    qrels_path = f"results/qrels_{args.corpus}_{args.model}.txt"
    with open(qrels_path, 'w') as f:
        f.write('\n'.join(qrels_output))

    print(f"Qrels-like output saved to {qrels_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)