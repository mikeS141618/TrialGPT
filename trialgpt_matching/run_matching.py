__author__ = "qiao"

"""
Running the TrialGPT matching for three cohorts (sigir, TREC 2021, TREC 2022).
"""

import argparse
import json
import os

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from TrialGPT import trialgpt_matching


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TrialGPT matching for specified corpus and model.")
    parser.add_argument("corpus", help="Corpus name")
    parser.add_argument("model", help="Model to use for matching")
    parser.add_argument("overwrite", help="Overwrite existing results (true/false)")
    return parser.parse_args()


def main(args):
    dataset = json.load(open(f"results/retrieved_trials.json"))
    output_path = f"results/matching_results_{args.corpus}_{args.model}.json"

    # Dict{Str(patient_id): Dict{Str(label): Dict{Str(trial_id): Str(output)}}}
    if args.overwrite.lower() == 'true' or not os.path.exists(output_path):
        print(f"Creating new matching results for {args.corpus} with {args.model}")
        output = {}
    else:
        print(f"Loading existing matching results for {args.corpus} with {args.model}")
        output = json.load(open(output_path))

    # Outer progress bar for patients
    for instance in tqdm(dataset, desc="Processing patients", unit="patient"):
        patient_id = instance["patient_id"]
        patient = instance["patient"]
        sents = sent_tokenize(patient)
        sents.append(
            "The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
        sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
        patient = "\n".join(sents)

        if patient_id not in output:
            output[patient_id] = {"0": {}, "1": {}, "2": {}}

        # Middle progress bar for labels
        for label in tqdm(["2", "1", "0"], desc=f"Labels for patient {patient_id}", leave=False):
            if label not in instance: continue

            # Inner progress bar for trials
            for trial in tqdm(instance[label], desc=f"Trials for label {label}", leave=False):
                trial_id = trial["NCTID"]

                if args.overwrite.lower() != 'true' and trial_id in output[patient_id][label]:
                    continue

                try:
                    results = trialgpt_matching(trial, patient, args.model)
                    output[patient_id][label][trial_id] = results

                    # Save after each trial to ensure progress is not lost
                    with open(output_path, "w") as f:
                        json.dump(output, f, indent=4)

                except Exception as e:
                    print(f"Error processing trial {trial_id} for patient {patient_id}: {e}")
                    continue

    print(f"Matching results saved to {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)