
import pandas as pd
import numpy as np
import statsmodels.api as sm
from data_generator import create_sample_data
from computingstrat import compute_perf_stats

#STEP 1: DONE
# I take fake_data from data_generator.py using create_sample_data fucntion.
# then I try to use compute_perf_stats from ComputingStrat.py

fake_data = create_sample_data(days=252, initial_aum=100000, ret_mean=0.001, ret_std=0.02, 
                      spy_mean=0.001, spy_std=0.015)
# Run the analysis

stats = compute_perf_stats(fake_data)
