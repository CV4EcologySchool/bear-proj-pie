# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import numpy as np
import shutil
import os
import os.path as osp
import imageio
from skimage import transform
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ['visualize_ranked_results']

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(
    distmat, query, width=128, height=256, save_dir='', topk=10, resize=True
):
    """Visualizes ranked results.
    Ranks will be plotted in a single figure.

    Args:
        distmat (numpy.ndarray): dist matrix of shape (num_query, num_query).
        query (list of tuples): a query set which is a list of tuples
                    of (img_path(s), pid, camid, dsetid).
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be
                    visualized. Default is 10.
    """
    assert distmat.shape[0] == distmat.shape[1]
    print('distmat shape', distmat.shape)
    num_q = distmat.shape[0]
    os.makedirs(save_dir, exist_ok=True)

    print('# query: {}'.format(num_q))
    print('Visualizing top-{} ranks ...'.format(topk))
    
    print('query length: {}'.format(len(query)))
    assert num_q == len(query)

    indices = np.argsort(distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            os.makedirs(dst, exist_ok=True)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(
                dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src)
            )
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid = query[q_idx]['impath'], query[q_idx]['pid']
        qimg_path_name = qimg_path

        qimg = imageio.imread(qimg_path)
        if resize:
            qimg = transform.resize(qimg, (width, height), order=3, anti_aliasing=True)
        ncols = topk + 1
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        ax[0].imshow(qimg)
        ax[0].set_title('Query {}'.format(qpid[:25]))
        ax[0].axis('off')

        rank_idx = 1
        for g_idx in indices[q_idx, 1:]:
            gimg_path, gpid = query[g_idx]['impath'], query[g_idx]['pid']

            matched = gpid == qpid
            border_color = 'green' if matched else 'red'
            gimg = imageio.imread(gimg_path)
            if resize:
                gimg = transform.resize(
                    gimg, (width, height), order=3, anti_aliasing=True
                )

            ax[rank_idx].imshow(gimg)
            ax[rank_idx].set_title('{}'.format(gpid[:25]))
            ax[rank_idx].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False,
                right=False,
                left=False,
                labelleft=False,
            )
            for loc, spine in ax[rank_idx].spines.items():
                spine.set_color(border_color)
                spine.set_linewidth(BW)

            rank_idx += 1
            if rank_idx > topk:
                break

        # Save figure
        fig_name = osp.basename(osp.splitext(qimg_path_name)[0])
        fig_path = osp.join(save_dir, fig_name + '.jpg')
        fig.savefig(fig_path, format='jpg', dpi=100, bbox_inches='tight', facecolor='w')
        plt.close(fig)

        if (q_idx + 1) % 10 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

        if q_idx >= 100:
            break

    print('Done. Images have been saved to "{}" ...'.format(save_dir))
