#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import os
import sys

import parse
import dist


def main():
    args = parseArgs(sys.argv)
    WT = parse.embeddings(args.embeddings)
    WTn = np.linalg.norm(WT, axis=1)
    # get analogy files in the analogy directory
    # http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory-in-python
    anlgyFns = [os.path.join(args.analogies, fn) for fn in
                next(os.walk(args.analogies))[2]]
    for anlgyFn in anlgyFns:
        print('Working on {:s}'.format(anlgyFn))
        with open(anlgyFn, 'rb') as anlgyF:
            with open(os.path.join(args.output, os.path.basename(anlgyFn)),
                      'w') as outF:
                for anlgy in anlgyF:
                    words = anlgy.split()
                    words.append(
                        args.distance_measure(WT, WTn, words[0], words[1],
                                              words[2]))
                    outF.write(' '.join(words) + '\n')


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description='Word analogy solver. Written in Python 2.7.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('embeddings',
                        type=validFile,
                        help='The word embeddings file.')
    parser.add_argument('analogies',
                        type=validDirectory,
                        help='The directory containing analogy files.')
    parser.add_argument('output',
                        type=validDirectory,
                        help='The directory for output files.')
    parser.add_argument('-d', '--distance_measure',
                        default='cosadd',
                        type=distMeasure,
                        help='''The distance measure to use.
                             {"cosadd", "cosmult"}''')
    return parser.parse_args()


def validFile(fileName):
    if os.path.isfile(fileName):
        return fileName
    msg = 'File "{:s}" does not exist.'.format(fileName)
    raise argparse.ArgumentTypeError(msg)


def validDirectory(dirName):
    if os.path.isdir(dirName):
        return dirName
    msg = 'Directory "{:s}" does not exist.'.format(dirName)
    raise argparse.ArgumentTypeError(msg)


def distMeasure(dm):
    if dm == 'cosadd':
        return dist.cosadd
    if dm == 'cosmult':
        return dist.cosmult
    msg = 'Distance measuere "{:s}" is not defined.'.format(dm)
    raise argparse.ArgumentTypeError(msg)


if __name__ == '__main__':
    main()
