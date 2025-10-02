#!/bin/bash

# This command ensures that the script will exit immediately if any command fails.
set -e

echo "--- STARTING COMPLETE EMPIRICAL EXPERIMENT BATCH ---"

# =============================================================
# SLC EMPIRICAL EXPERIMENTS (8 total)
# =============================================================
echo "\n[1/6] Running: SLC Empirical, Additive Gaussian Noise..."
python run_empirical_evaluation.py --experiment slc --shift_type additive --distribution gaussian --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20

# echo "\n[2/16] Running: SLC Empirical, Multiplicative Gaussian Noise..."
# python run_empirical_evaluation.py --experiment slc --shift_type multiplicative --distribution gaussian --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20

echo "\n[2/6] Running: SLC Empirical, Additive Student-t Noise..."
python run_empirical_evaluation.py --experiment slc --shift_type additive --distribution student-t --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20

# echo "\n[4/16] Running: SLC Empirical, Multiplicative Student-t Noise..."
# python run_empirical_evaluation.py --experiment slc --shift_type multiplicative --distribution student-t --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20

echo "\n[3/6] Running: SLC Empirical, Additive Exponential Noise..."
python run_empirical_evaluation.py --experiment slc --shift_type additive --distribution exponential --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20

# echo "\n[6/16] Running: SLC Empirical, Multiplicative Exponential Noise..."
# python run_empirical_evaluation.py --experiment slc --shift_type multiplicative --distribution exponential --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20

# echo "\n[7/16] Running: SLC Empirical, Translation Shift..."
# python run_empirical_evaluation.py --experiment slc --shift_type additive --distribution translation --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20

# echo "\n[8/16] Running: SLC Empirical, Scaling Shift..."
# python run_empirical_evaluation.py --experiment slc --shift_type multiplicative --distribution scaling --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20


# =============================================================
# LILUCAS EMPIRICAL EXPERIMENTS (8 total)
# =============================================================
echo "\n[4/6] Running: LiLUCAS Empirical, Additive Gaussian Noise..."
python run_empirical_evaluation.py --experiment lilucas --shift_type additive --distribution gaussian --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20

# echo "\n[10/16] Running: LiLUCAS Empirical, Multiplicative Gaussian Noise..."
# python run_empirical_evaluation.py --experiment lilucas --shift_type multiplicative --distribution gaussian --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20

echo "\n[5/6] Running: LiLUCAS Empirical, Additive Student-t Noise..."
python run_empirical_evaluation.py --experiment lilucas --shift_type additive --distribution student-t --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20

# echo "\n[12/16] Running: LiLUCAS Empirical, Multiplicative Student-t Noise..."
# python run_empirical_evaluation.py --experiment lilucas --shift_type multiplicative --distribution student-t --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20

echo "\n[7/6] Running: LiLUCAS Empirical, Additive Exponential Noise..."
python run_empirical_evaluation.py --experiment lilucas --shift_type additive --distribution exponential --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20

# echo "\n[14/16] Running: LiLUCAS Empirical, Multiplicative Exponential Noise..."
# python run_empirical_evaluation.py --experiment lilucas --shift_type multiplicative --distribution exponential --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20

# echo "\n[15/16] Running: LiLUCAS Empirical, Translation Shift..."
# python run_empirical_evaluation.py --experiment lilucas --shift_type additive --distribution translation --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20

# echo "\n[16/16] Running: LiLUCAS Empirical, Scaling Shift..."
# python run_empirical_evaluation.py --experiment lilucas --shift_type multiplicative --distribution scaling --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 20


echo "\n--- EMPIRICAL EXPERIMENT BATCH COMPLETE ---" 