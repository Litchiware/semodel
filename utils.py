import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def grid_plot(images, titles, filename):
    assert len(images.shape) == 3
    m = len(titles)
    n = images.shape[0] / m
    m1 = int(np.ceil(m / 2.))

    fig = plt.figure()
    axes = []
    gs1 = gridspec.GridSpec(n, m1)
    gs1.update(top=0.95,
               bottom=0.55,
               left=0.05,
               right=0.85,
               hspace=0.05,
               wspace=0.15)
    axes += [fig.add_subplot(gs1[i, j]) for j in range(m1) for i in range(n)]

    gs2 = gridspec.GridSpec(n, m1)
    gs2.update(top=0.45,
               bottom=0.05,
               left=0.05,
               right=0.85,
               hspace=0.05,
               wspace=0.15)
    axes += [fig.add_subplot(gs2[i, j]) for j in range(m - m1) for i in range(n)]

    vmin, vmax = np.min(images), np.max(images)

    for i in range(len(axes)):
        im = axes[i].imshow(images[i], vmin=vmin, vmax=vmax)
        axes[i].set_aspect('equal')
        axes[i].xaxis.set_ticks([])
        axes[i].yaxis.set_ticks([])
        if i % n == 0:
            axes[i].set_title(titles[i/n])

    fig.add_subplot(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(filename)
