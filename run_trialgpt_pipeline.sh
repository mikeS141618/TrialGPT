#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Checkpoint file
CHECKPOINT_FILE="results/trialgpt_checkpoint.txt"

# Function to measure execution time
measure_time() {
    start_time=$(date +%s)
    "$@"
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Time taken: $duration seconds"
}

# Function to update checkpoint
update_checkpoint() {
    echo "$1" > "$CHECKPOINT_FILE"
}

# Function to get last completed step
get_last_step() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        cat "$CHECKPOINT_FILE"
    else
        echo "0"
    fi
}

# Set variables
CORPUS="sigir"
MODEL="gpt-4o" #"gpt-4o" "gpt-4-turbo" "gpt-4o-mini"
K=100
BM25_WEIGHT=1
MEDCPT_WEIGHT=1

echo "Starting TrialGPT pipeline..."

# Get the last completed step
LAST_STEP=$(get_last_step)

# Step 1: TrialGPT-Retrieval - Keyword Generation
if [ "$LAST_STEP" -lt 1 ]; then
    echo "Step 1: Keyword Generation"
    measure_time python trialgpt_retrieval/keyword_generation.py $CORPUS $MODEL
    update_checkpoint "1"
else
    echo "Skipping Step 1: Already completed"
fi

# Step 2: TrialGPT-Retrieval - Hybrid Fusion Retrieval
if [ "$LAST_STEP" -lt 2 ]; then
    echo "Step 2: Hybrid Fusion Retrieval"
    measure_time python trialgpt_retrieval/hybrid_fusion_retrieval.py $CORPUS $MODEL $K $BM25_WEIGHT $MEDCPT_WEIGHT
    update_checkpoint "2"
else
    echo "Skipping Step 2: Already completed"
fi

# Step 3: TrialGPT-Matching
if [ "$LAST_STEP" -lt 3 ]; then
    echo "Step 3: TrialGPT-Matching"
    measure_time python trialgpt_matching/run_matching.py $CORPUS $MODEL
    update_checkpoint "3"
else
    echo "Skipping Step 3: Already completed"
fi

# Step 4: TrialGPT-Ranking - Aggregation
if [ "$LAST_STEP" -lt 4 ]; then
    echo "Step 4: TrialGPT-Ranking Aggregation"
    MATCHING_RESULTS="results/matching_results_${CORPUS}_${MODEL}.json"
    measure_time python trialgpt_ranking/run_aggregation.py $CORPUS $MODEL $MATCHING_RESULTS
    update_checkpoint "4"
else
    echo "Skipping Step 4: Already completed"
fi

# Step 5: TrialGPT-Ranking - Final Ranking
if [ "$LAST_STEP" -lt 5 ]; then
    echo "Step 5: TrialGPT-Ranking Final Ranking"
    MATCHING_RESULTS="results/matching_results_${CORPUS}_${MODEL}.json"
    AGGREGATION_RESULTS="results/aggregation_results_${CORPUS}_${MODEL}.json"
    measure_time python trialgpt_ranking/rank_results.py $MATCHING_RESULTS $AGGREGATION_RESULTS
    update_checkpoint "5"
else
    echo "Skipping Step 5: Already completed"
fi

echo "TrialGPT pipeline completed successfully!"
# Clear checkpoint after successful completion
# rm -f "$CHECKPOINT_FILE"