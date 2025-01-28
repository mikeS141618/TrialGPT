__author__ = "qiao"

"""
Rank the trials given the matching and aggregation results
"""

import json
import os
import sys
import traceback

eps = 1e-9


def get_matching_score(matching):
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
    try:
        rel_score = float(assessment["relevance_score_R"])
        eli_score = float(assessment["eligibility_score_E"])
    except:
        rel_score = 0
        eli_score = 0

    score = (rel_score + eli_score) / 100

    return score

if __name__ == "__main__":
    # args are the results paths
    matching_results_path = sys.argv[1]
    agg_results_path = sys.argv[2]

    # loading the results
    matching_results = json.load(open(matching_results_path))
    agg_results = json.load(open(agg_results_path))

    # Create a directory for output if it doesn't exist
    output_dir = "results/trial_rankings"
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store all rankings
    all_rankings = {}

    # loop over the patients
    for patient_id, label2trial2results in matching_results.items():
        trial2score = {}
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

                    trial2score[trial_id] = trial_score

            sorted_trial2score = sorted(trial2score.items(), key=lambda x: -x[1])

            print()
            print(f"Patient ID: {patient_id}")
            print("Clinical trial ranking:")

            # Save to individual text file
            output_file = os.path.join(output_dir, f"trialranking_{patient_id}.txt")
            with open(output_file, 'w') as f:
                f.write(f"Patient ID: {patient_id}\n")
                f.write("Clinical trial ranking:\n")
                for trial, score in sorted_trial2score:
                    line = f"{trial}: {score}\n"
                    f.write(line)

            print("===")
            print()
            print(f"Ranking saved to {output_file}")

            # Add to all_rankings dictionary
            all_rankings[patient_id] = dict(sorted_trial2score)

        except Exception as e:
            print(f"An error occurred while processing patient {patient_id}:")
            print(traceback.format_exc())
            print("Continuing with the next patient...")
            print()

    # Save all rankings to a single JSON file
    all_rankings_file = os.path.join(output_dir, "all_rankings.json")
    with open(all_rankings_file, 'w') as f:
        json.dump(all_rankings, f, indent=2)

    print(f"All rankings saved to {all_rankings_file}")

