#!/bin/bash

# Exit immediately if a command exits with a non-zero status
# set -e

# Add this at the beginning of your script
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Checkpoint file
CHECKPOINT_FILE="results/trialgpt_checkpoint.txt"

# Function to measure execution time
measure_time() {
    start_time=$(date +%s)
    "$@"
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Time taken: $duration seconds"
    return $duration
}

# Function to update checkpoint
update_checkpoint() {
    local step=$1
    local duration=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "Step $step completed at $timestamp (Duration: ${duration}s)" >> "$CHECKPOINT_FILE"
}

# Function to get last completed step
get_last_step() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        tail -n 1 "$CHECKPOINT_FILE" | cut -d' ' -f2
    else
        echo "0"
    fi
}

# Set variables
CORPUS="sigir"
Q_TYPE="gpt-4o-mini"
K=20
BM25_WEIGHT=1
MEDCPT_WEIGHT=1
OVER_WRITE='true'
TOP_K=100
BATCH_SIZE=512
PERCENTILE1=90
PERCENTILE2=70

echo "Starting TrialGPT pipeline..."

# Get the last completed step
LAST_STEP=$(get_last_step)

# Step 1: TrialGPT-Retrieval - Keyword Generation
if [ "$LAST_STEP" -lt 1 ]; then
    echo "Step 1: Keyword Generation"
    measure_time python trialgpt_retrieval/keyword_generation.py $CORPUS $Q_TYPE
    duration=$?
    update_checkpoint "1" $duration
else
    echo "Skipping Step 1: Already completed"
fi

# Step 2: TrialGPT-Retrieval - Hybrid Fusion Retrieval and Generate Retrieved Trials
if [ "$LAST_STEP" -lt 2 ]; then
    echo "Step 2: Hybrid Fusion Retrieval and Generate Retrieved Trials"
    measure_time python trialgpt_retrieval/hybrid_fusion_retrieval.py $CORPUS $Q_TYPE $K $BM25_WEIGHT $MEDCPT_WEIGHT $OVER_WRITE $TOP_K $BATCH_SIZE $PERCENTILE1 $PERCENTILE2
    duration=$?
    update_checkpoint "2" $duration
else
    echo "Skipping Step 2: Already completed"
fi

## Step 3: TrialGPT-Matching
#if [ "$LAST_STEP" -lt 3 ]; then
#    echo "Step 3: TrialGPT-Matching"
#    measure_time python trialgpt_matching/run_matching.py $CORPUS $Q_TYPE $OVER_WRITE
#    duration=$?
#    update_checkpoint "3" $duration
#else
#    echo "Skipping Step 3: Already completed"
#fi
#
## Step 4: TrialGPT-Ranking - Aggregation
#if [ "$LAST_STEP" -lt 4 ]; then
#    echo "Step 4: TrialGPT-Ranking Aggregation"
#    MATCHING_RESULTS="results/matching_results_${CORPUS}_${Q_TYPE}.json"
#    measure_time python trialgpt_ranking/run_aggregation.py $CORPUS $Q_TYPE $MATCHING_RESULTS $OVER_WRITE
#    duration=$?
#    update_checkpoint "4" $duration
#else
#    echo "Skipping Step 4: Already completed"
#fi
#
## Step 5: TrialGPT-Ranking - Final Ranking
#if [ "$LAST_STEP" -lt 5 ]; then
#    echo "Step 5: TrialGPT-Ranking Final Ranking"
#    MATCHING_RESULTS="results/matching_results_${CORPUS}_${Q_TYPE}.json"
#    AGGREGATION_RESULTS="results/aggregation_results_${CORPUS}_${Q_TYPE}.json"
#    measure_time python trialgpt_ranking/rank_results.py $MATCHING_RESULTS $AGGREGATION_RESULTS $OVER_WRITE $CORPUS $Q_TYPE
#    duration=$?
#    update_checkpoint "5" $duration
#else
#    echo "Skipping Step 5: Already completed"
#fi
#
#echo "TrialGPT pipeline completed successfully!"
#echo "Checkpoint file contents:"
#cat "$CHECKPOINT_FILE"
