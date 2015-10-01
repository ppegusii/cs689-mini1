#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys


def main():
    args = parseArgs(sys.argv)
    # get analogy files in the analogy directory
    # http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory-in-python
    resultFns = [os.path.join(args.results, fn) for fn in
                 next(os.walk(args.results))[2]]
    accSum = 0.0
    resultsCnt = 0
    for resultFn in resultFns:
        lineCnt = 0
        corrCnt = 0
        with open(resultFn, 'rb') as resultF:
            for result in resultF:
                lineCnt += 1
                words = result.split()
                if len(words) < 5:
                    break
                if words[3] == words[4]:
                    corrCnt += 1
            if lineCnt < 1:
                continue
            acc = float(corrCnt) / lineCnt
            print('{:s} {:0.2f}'.format(
                os.path.basename(resultFn),
                acc))
        accSum += acc
        resultsCnt += 1
    print('Average {:0.2f}'.format(accSum / resultsCnt))


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description='Evaluate analogy results. Written in Python 2.7.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results',
                        type=validDirectory,
                        help='The directory containing result files.')
    return parser.parse_args()


def validDirectory(dirName):
    if os.path.isdir(dirName):
        return dirName
    msg = 'Directory "{:s}" does not exist.'.format(dirName)
    raise argparse.ArgumentTypeError(msg)


if __name__ == '__main__':
    main()
