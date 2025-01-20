import numpy as np

experiments = {
                1: 'synth1',
                2: 'synth1_gnd',
                2: 'lucas'
              }

n_envs      = {
                'synth1': [3, 3],
                'synth1_gnd': [3, 2],
                'lucas': [2, 2]
              }

n_samples   = {
                'synth1': [100000, 100000],
                'synth1_gnd': [100000, 100000],
                'lucas': [100000, 100000]
              }

radius      = {
                'synth1': [0.5, 0.3],
                'synth1_gnd': [0.5, 0.3],
                'lucas': [0.8, 0.6]
              }