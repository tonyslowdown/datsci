#!/usr/bin/env python
'''
Description     : Utilities for feature engineering
Author          : Jin Kim jjinking(at)gmail(dot)com
License         : MIT
Creation date   : 2016.01.28
Last Modified   : 2015.01.28
Modified By     : Jin Kim jjinking(at)gmail(dot)com
'''

import pandas as pd
from sklearn.decomposition import PCA


def biplot(df):
    '''
    Create biplot using principle components 1 and 2
    Play around with the ranges for scaling the plot
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1, 3])
    '''

    # Fit on 2 components
    pca = PCA(n_components=2, whiten=True).fit(df)

    # Plot transformed/projected data
    ax = pd.DataFrame(
        pca.transform(df),
        columns=['PC1', 'PC2']
    ).plot(kind='scatter', x='PC1', y='PC2', figsize=(10, 8), s=0.8)

    # Plot arrows and labels
    pc1_pc2 = zip(pca.components_[0], pca.components_[1])
    for i, (pc1, pc2) in enumerate(pc1_pc2):
        ax.arrow(0, 0, pc1, pc2, width=0.001, fc='orange', ec='orange')
        ax.annotate(df.columns[i], (pc1, pc2), size=12)

    return ax
