import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

col_names = pd.read_csv('data/names.csv')  # 'data/names.csv'
data = pd.read_csv('data/breast-cancer-wisconsin.data', names=col_names.columns)
data = data[data["bare_nuclei"] != '?']

#The distributions associated with each feature are... eh
data.groupby('class').hist(figsize=(12, 12))

#print correlations
corr = data.corr()
print(corr)

#I didn't write this function: all credit for this function goes to borisbanushev and can be found here: https://github.com/borisbanushev/predictions/blob/master/predictions.ipynb
#I just really like this function and have been consistently using it for months now, big ups
def diagonal_correlation_matrix():
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


plt.figure()
diagonal_correlation_matrix()
plt.show()

