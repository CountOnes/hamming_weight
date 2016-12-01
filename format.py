#!/usr/bin/env python

import sys

def format(file):
    for line in file:
        line = line.strip()
        try:
            pos = line.index(':')
        except ValueError:
            print line
            continue

        # a measurment
        F = line[pos + 1:].split()
        fun  = line[:line.index('(')] # trim the signature

        if 'error' in line:
            print '%-50s : FAILED !!!' % fun
        else:
            best = float(F[0])
            avg  = float(F[5])

            print '%-50s : %6.2f %6.2f' % (fun, best, avg)

if __name__ == '__main__':
    format(sys.stdin)
