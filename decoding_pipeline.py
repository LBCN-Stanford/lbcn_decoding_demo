# Import Libraries
from initDirs import dirs
from prepDataDecoding import prepDataDecoding
from decoding_schemes import Decoding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Subjects
subject = 'S13_57_TVD'
# Project name
project_name = 'MMR'
# Conditions to decode
conditions = [['math_memory', 'math_memory']]

# Additional processing params
baselinecorr = 'nobaseline'
dec_method = 'LogisticRegression'
dec_scorer = 'accuracy'
gatordiag = 'gat'
decimate = 10

# Prepare data
params = prepDataDecoding(dirs, subject, project_name, conditions[0][0], conditions[0][1], baselinecorr, decimate)
# Run the decoding and retrieve the scores
scores = Decoding(params, dec_method, dec_scorer, gatordiag)
# Update times, since data was decimated
times = np.linspace(-0.5,5,np.size(params['times']))


# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
ax1.plot(times, np.diag(scores), label='score', color='k')
ax1.axhline(.5, color='k', linestyle=':', label='chance')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('accuracy (%)')
ax1.legend()
ax1.axvline(.0, color='k', linestyle='-')
ax1.set_title('Decoding iEEG electrodes over time')

# Plot the full matrix
im = ax2.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r', extent=times[[0, -1, 0, -1]])
ax2.set_xlabel('Testing Time (s)')
ax2.set_ylabel('Training Time (s)')
ax2.set_title('Temporal Generalization')
ax2.axvline(0, color='k')
ax2.axhline(0, color='k')
cb = plt.colorbar(im, ax=ax2)
cb.set_label('accuracy (%)')

plt.show()