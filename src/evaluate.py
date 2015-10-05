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
    glbCorrCnt = 0
    glbLineCnt = 0
    semCorrCnt = 0
    semLineCnt = 0
    synCorrCnt = 0
    synLineCnt = 0
    for resultFn in resultFns:
        lineCnt = 0
        corrCnt = 0
        ansInQuCnt = 0
        with open(resultFn, 'rb') as resultF:
            # print(resultFn)
            for result in resultF:
                words = result.split()
                # print(words)
                if words[4] == '**NO_EMBEDDING_FOR_A_WORD**':
                    continue
                if len(words) < 5:
                    break
                if words[3] == words[4]:
                    corrCnt += 1
                if words[4] in words[:3]:
                    ansInQuCnt += 1
                lineCnt += 1
            if lineCnt < 1:
                continue
            acc = float(corrCnt) / lineCnt
            ansInQu = float(ansInQuCnt) / lineCnt
            print('{:s} {:0.3f} {:0.3f}'.format(
                os.path.basename(resultFn),
                acc, ansInQu))
        glbLineCnt += lineCnt
        glbCorrCnt += corrCnt
        if os.path.basename(resultFn).startswith('gram'):
            synCorrCnt += corrCnt
            synLineCnt += lineCnt
        else:
            semCorrCnt += corrCnt
            semLineCnt += lineCnt
    if semLineCnt > 0:
        print('Sem Av {:0.3f}'.format(float(semCorrCnt) / semLineCnt))
    if synLineCnt > 0:
        print('Syn Av {:0.3f}'.format(float(synCorrCnt) / synLineCnt))
    print('Average {:0.3f}'.format(float(glbCorrCnt) / glbLineCnt))


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
