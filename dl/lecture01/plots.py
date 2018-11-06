from operator import itemgetter
from textwrap import wrap

import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(image, predictions, best_color='darkorange',
                     other_colors='royalblue', sort_by_probability=False,
                     edge_color='white', alpha=0.8, title=None,
                     wrap_text=None, labels_font_size=18, ax=None,
                     **plot_params):

    if not plot_params:
        plot_params['figsize'] = (10, 8)

    if isinstance(image, str):
        image = plt.imread(image)

    sort_key = 1 if sort_by_probability else 0
    labels, values = zip(
        *sorted(predictions.items(), key=itemgetter(sort_key), reverse=True))
    n = len(labels)
    colors = [other_colors] * n
    max_index = np.argmax(values)
    colors[max_index] = best_color

    if not sort_by_probability:
        labels, values, colors = [
            list(reversed(x)) for x in (labels, values, colors)]

    w, h, _ = image.shape
    scaled_values = [w*v for v in values]
    bar_height = 0.5 * h / n
    extent = [0, w, 0, h]

    if ax is None:
        f, ax = plt.subplots(1, 1, **plot_params)

    y_pos = [(i + 1)*h/(n + 1) for i in np.arange(n)]
    ax.barh(y_pos, scaled_values, height=bar_height,
            edgecolor=edge_color, align='center', color=colors, alpha=alpha)

    if wrap_text is not None:
        labels = ['\n'.join(wrap(label, width=wrap_text)) for label in labels]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=labels_font_size)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.invert_yaxis()
    ax.imshow(image, zorder=0, extent=extent)
    if title is not None:
        ax.set_title(title)

    return ax
