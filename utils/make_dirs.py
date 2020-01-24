import os
import numpy as np
from glob import glob

input_dir = 'data/raw'
sub_dirs = ['positive', 'negative']
out_dir = 'data'
samples = dict()

# Get a set of negative and positive samples
for i in sub_dirs:
    target_dir = os.path.join(input_dir, i)
    samples[i] = set()
    for j in os.listdir(target_dir):
        samples[i].add('_'.join(j.split('_', 3)[:-1]))

# Split into training validation and test
# 60% - train set,
# 20% - validation set,
# 20% - test set
for k in samples.keys():
    df = list(samples[k])
    train, validate, test = np.split(np.random.choice(df,
                                                      df.__len__(),
                                                      replace=False),
                                     [int(.6 * len(df)), int(.8 * len(df))])
    target_dir = os.path.join(input_dir, k)
    t1 = 0
    for l in train:
        for o in glob(os.path.join(target_dir, l + '*jpg')):
            t1 += 1
            print('t1 = {}\tl = {}'.format(t1, l), end="\r", flush=True)
            os.link(o, os.path.join(out_dir, 'train', k, os.path.basename(l)))

    print('\nBeginning validation')
    t2 = 0
    for m in validate:
        for p in glob(os.path.join(target_dir, m + '*jpg')):
            t2 += 1
            print('t2 = {}\tl = {}'.format(t2, m), end="\r", flush=True)
            os.link(p, os.path.join(out_dir, 'validation', os.path.basename(m)))

    print('\nBeginning test')
    t3 = 0
    for n in test:
        for q in glob(os.path.join(target_dir, n + '*jpg')):
            t3 += 1
            print('t3 = {}\tl = {}'.format(t3, n), end="\r", flush=True)
            os.link(q, os.path.join(out_dir, 'test', k, os.path.basename(m)))

    print('\nFor {}, completed {} training, {} validation, and {} test'.format(k, t1, t2, t3))
