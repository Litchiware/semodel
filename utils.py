import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def grid_axes(nSamples, nClasses, nGrid, fig, cbar_ax=False):
    import matplotlib.gridspec as gridspec

    nrows = nSamples / nClasses
    ncols = int(np.ceil(nClasses/float(nGrid)))
    h = 1 / (nGrid * (1.05 * nrows + 0.45))
    gs = []
    for k in range(nGrid):
        g = gridspec.GridSpec(nrows, ncols)
        top = 1 - 0.25 * h - k * (nrows * h + (nrows - 1) * 0.05 * h + 0.5 * h)
        bottom = top - (nrows * h + (nrows - 1) * 0.05 * h + 0.5 * h) + 0.5 * h
        if cbar_ax:
            left = 0.05
            right = 0.85
        else:
            left = 0.1
            right = 0.9
        g.update(top = top,
                 bottom = bottom,
                 left = left,
                 right = right,
                 hspace = 0.05,
                 wspace = 0.15)
        gs.append(g)

    gs = [g[i, j] for g in gs for j in range(ncols) for i in range(nrows)]
    if nGrid * ncols > nClasses:
        del gs[nrows*(nClasses-nGrid*ncols):]
        
    fig.clf()
    axes = [fig.add_subplot(g) for g in gs]

    if cbar_ax:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
        return axes, cbar_ax

    return axes


def gplot_images(samples, titles, filename, ngrid=2, figsize=(12, 8)):
    assert len(samples.shape) == 3, "array of images should have 3 dims"

    fig = plt.figure(figsize=figsize)
    axes, cbar_ax = grid_axes(len(samples), len(titles), ngrid, fig, cbar_ax=True)
    nrows = len(samples) / len(titles)

    vmin, vmax = np.min(samples), np.max(samples)
    for ax, (i, sample) in zip(axes, enumerate(samples)):
        if i % nrows == 0:
            ax.set_title(titles[i/nrows])

        im = ax.imshow(sample, vmin=vmin, vmax=vmax, aspect='auto')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(filename)

def gplot_lines(samples, titles, filename, ngrid=2, figsize=(12, 8)):
    assert len(samples.shape) == 2, "array of lines should have 2 dims"

    fig = plt.figure(figsize=figsize)
    axes = grid_axes(len(samples), len(titles), ngrid, fig)
    nrows = len(samples) / len(titles)

    vmin, vmax = np.min(samples), np.max(samples)
    for ax, (i, sample) in zip(axes, enumerate(samples)):
        if i % nrows == 0:
            ax.set_title(titles[i/nrows])

        ax.plot(sample)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_xlim([0, sample.shape[0]-1])
        ax.set_ylim([vmin, vmax])

    plt.savefig(filename)

if __name__ == '__main__':
    images = np.array([np.eye(2)]*45)
    titles = [str(i) for i in range(15)]
    filename = "test-images.png"
    gplot_images(images, titles, filename, ngrid=2)

    lines = np.array([[1,2,3,4]]*45)
    titles = [str(i) for i in range(15)]
    filename = "test-lines.png"
    gplot_lines(lines, titles, filename, ngrid=2)
