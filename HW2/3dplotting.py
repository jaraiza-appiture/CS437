import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

def plot_3d(D,Labels,K):
    tracers_data = []
    Z = []
    for k in range(K):
        Z.append([D[n] for n in range(len(D)) if Labels[n] == k])
    for cluster in Z:
        tracers_data.append(
            go.Scatter3d(
                x = [row[0] for row in cluster],
                y = [row[1] for row in cluster],
                z = [row[2] for row in cluster],
                mode='makers',
                marker=dict(
                    color='rgb(%d,%d,%d)'%(np.random.randint(257),
                                           np.random.randint(257),
                                           np.random.randint(257)),
                    size=5,
                    symbol='circle',
                    line=dict(
                        color='rgb(%d,%d,%d)'%(np.random.randint(257),
                                               np.random.randint(257),
                                               np.random.randint(257)),
                        width=0.5
                    ),
                    opacity=0.8
                )
            )
        )
    layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
        )
    )

    fig = go.Figure(data=tracers_data, layout=layout)
    py.iplot(fig, filename='Kmeans-3d-scatter')
