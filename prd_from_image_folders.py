# coding=utf-8
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import hashlib

import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
from inception_torch import InceptionV3
import prd_score as prd
import prd_score_classifier as prdc


parser = argparse.ArgumentParser(
    description='Assessing Generative Models via Precision and Recall',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--reference_dir', type=str, required=True,
                    help='directory containing reference images')
parser.add_argument('--eval_dirs', type=str, nargs='+', required=True,
                    help='directory or directories containing images to be '
                    'evaluated')
parser.add_argument('--eval_labels', type=str, nargs='+', required=True,
                    help='labels for the eval_dirs (must have same size)')
parser.add_argument('--num_clusters', type=int, default=20,
                    help='number of cluster centers to fit')
parser.add_argument('--num_angles', type=int, default=1001,
                    help='number of angles for which to compute PRD, must be '
                         'in [3, 1e6]')
parser.add_argument('--num_runs', type=int, default=10,
                    help='number of independent runs over which to average the '
                         'PRD data')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='maximal number of epochs for training the classifier')
parser.add_argument('--patience', type=int, default=4,
                    help='Early stopping patience parameter')
parser.add_argument('--plot_path', type=str, default=None,
                    help='path for final plot file (can be .png or .pdf)')
parser.add_argument('--cache_dir', type=str, default='/tmp/prd_cache/',
                    help='cache directory')
parser.add_argument('--classif', dest='classif', action='store_true',
                    help='enable the new implementation')
parser.add_argument('--raw', dest='use_raw_images', action='store_true',
                    help='enable the new implementation')
parser.add_argument('--silent', dest='verbose', action='store_false',
                    help='disable logging output')

args = parser.parse_args()

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda:0')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')


def generate_inception_embedding(imgs, layer_name='pool_3:0'):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    net = InceptionV3([block_idx], normalize_input=True)
    net = net.eval().to(device)
    imgs = imgs.to(device)
    i=0
    batch_size = 32
    embeddings = []
    while i < len(imgs):
        batch = imgs[i:i+batch_size]
        feat = net(batch).cpu().numpy()
        embeddings.append(feat)
        i += batch_size
    return np.concatenate(embeddings, axis=0)


def load_or_generate_inception_embedding(directory, cache_dir, use_raw_images=False):
    directory = os.path.realpath(directory)
    hash = hashlib.md5(directory.encode('utf-8')).hexdigest()
    path = os.path.join(cache_dir, hash + '.npy')
    if os.path.exists(path) and (os.stat(path).st_mtime > os.stat(directory).st_mtime) and not use_raw_images:
        embeddings = np.load(path)
        return embeddings
    imgs = load_images_from_dir(directory)
    if use_raw_images:
        imgs = imgs.reshape([imgs.shape[0],-1])
        return imgs.astype(np.float32)
    embeddings = generate_inception_embedding(imgs)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(path, 'wb') as f:
        np.save(f, embeddings)
    return embeddings


def load_images_from_dir(directory, types=('png', 'jpg', 'bmp', 'gif')):
    paths = [os.path.join(directory, fn) for fn in sorted(os.listdir(directory))
             if os.path.splitext(fn)[-1][1:] in types]
    from PIL import Image
    imgs = [torchvision.transforms.ToTensor()(Image.open(path)).unsqueeze(0) for path in paths]
    return torch.cat(imgs)


def cluster_main(reference_dir, eval_dirs):
    if args.verbose:
        print('computing inception embeddings for ' + reference_dir)
    real_embeddings = load_or_generate_inception_embedding(
        reference_dir, args.cache_dir, args.use_raw_images)
    prd_data = []
    for directory in eval_dirs:
        if args.verbose:
            print('computing inception embeddings for ' + directory)
        eval_embeddings = load_or_generate_inception_embedding(
            directory, args.cache_dir, args.use_raw_images)
        if args.verbose:
            print('computing PRD')
        prd_data.append(prd.compute_prd_from_embedding(
            eval_data=eval_embeddings,
            ref_data=real_embeddings,
            num_clusters=args.num_clusters,
            num_angles=args.num_angles,
            num_runs=args.num_runs))

    return prd_data


def classif_main(reference_dir, eval_dirs):
    prd_data = []
    for directory in eval_dirs:
        if args.verbose:
            print('computing PRD for' + directory)
        prd_data.append(prdc.computePRD(source_folder=directory,
                        target_folder=reference_dir,
                        num_angles=args.num_angles,
                        num_runs=args.num_runs, num_epochs=args.num_epochs,
                        patience=args.patience))

    return prd_data

if __name__ == '__main__':
    if len(args.eval_dirs) != len(args.eval_labels):
        raise ValueError(
            'Number of --eval_dirs must be equal to number of --eval_labels.')

    reference_dir = os.path.abspath(args.reference_dir)
    eval_dirs = [os.path.abspath(directory) for directory in args.eval_dirs]

    if args.classif:
        prd_data = classif_main(reference_dir, eval_dirs)
    else:
        prd_data = cluster_main(reference_dir, eval_dirs)

    if args.verbose:
        print('plotting results')

    print()
    f_beta_data = [prd.prd_to_max_f_beta_pair(precision, recall, beta=8)
                   for precision, recall in prd_data]
    print('F_8   F_1/8     model')
    for directory, f_beta in zip(eval_dirs, f_beta_data):
        print('%.3f %.3f     %s' % (f_beta[0], f_beta[1], directory))

    prd.plot(prd_data, labels=args.eval_labels, out_path=args.plot_path)
