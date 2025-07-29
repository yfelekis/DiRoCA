#!/bin/bash

# This command ensures that the script will exit immediately if any command fails.
set -e

echo "--- STARTING COMPLETE EXPERIMENT BATCH ---"

# =============================================================
# SLC EXPERIMENTS (8 total)
# =============================================================
echo "\n[1/16] Running: SLC, Additive Gaussian Noise..."
python run_evaluation.py --experiment slc --shift_type additive --distribution gaussian --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[2/16] Running: SLC, Multiplicative Gaussian Noise..."
python run_evaluation.py --experiment slc --shift_type multiplicative --distribution gaussian --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[3/16] Running: SLC, Additive Student-t Noise..."
python run_evaluation.py --experiment slc --shift_type additive --distribution student-t --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[4/16] Running: SLC, Multiplicative Student-t Noise..."
python run_evaluation.py --experiment slc --shift_type multiplicative --distribution student-t --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[5/16] Running: SLC, Additive Exponential Noise..."
python run_evaluation.py --experiment slc --shift_type additive --distribution exponential --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[6/16] Running: SLC, Multiplicative Exponential Noise..."
python run_evaluation.py --experiment slc --shift_type multiplicative --distribution exponential --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[7/16] Running: SLC, Translation Shift..."
python run_evaluation.py --experiment slc --shift_type additive --distribution translation --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[8/16] Running: SLC, Scaling Shift..."
python run_evaluation.py --experiment slc --shift_type multiplicative --distribution scaling --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20 --zero_mean False


# =============================================================
# LILUCAS EXPERIMENTS (8 total)
# =============================================================
echo "\n[9/16] Running: LiLUCAS, Additive Gaussian Noise..."
python run_evaluation.py --experiment lilucas --shift_type additive --distribution gaussian --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[10/16] Running: LiLUCAS, Multiplicative Gaussian Noise..."
python run_evaluation.py --experiment lilucas --shift_type multiplicative --distribution gaussian --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[11/16] Running: LiLUCAS, Additive Student-t Noise..."
python run_evaluation.py --experiment lilucas --shift_type additive --distribution student-t --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[12/16] Running: LiLUCAS, Multiplicative Student-t Noise..."
python run_evaluation.py --experiment lilucas --shift_type multiplicative --distribution student-t --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[13/16] Running: LiLUCAS, Additive Exponential Noise..."
python run_evaluation.py --experiment lilucas --shift_type additive --distribution exponential --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[14/16] Running: LiLUCAS, Multiplicative Exponential Noise..."
python run_evaluation.py --experiment lilucas --shift_type multiplicative --distribution exponential --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[15/16] Running: LiLUCAS, Translation Shift..."
python run_evaluation.py --experiment lilucas --shift_type additive --distribution translation --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20 --zero_mean False

echo "\n[16/16] Running: LiLUCAS, Scaling Shift..."
python run_evaluation.py --experiment lilucas --shift_type multiplicative --distribution scaling --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20 --zero_mean False


echo "\n--- EXPERIMENT BATCH COMPLETE ---"