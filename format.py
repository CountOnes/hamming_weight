#!/usr/bin/env python

import sys

def format(file):
    for line in file:
        line = line.strip()
        if ':' in line: # a measurment
            F = line.split()
            fun  = F[0]
            fun  = fun[:fun.index('(')] # trim the signature

            if 'error' in line:
                print '%-40s : FAILED !!!' % fun
            else:
                best = float(F[2])
                avg  = float(F[7])

                print '%-40s : %6.2f %6.2f' % (fun, best, avg)
        else:
            print line

if __name__ == '__main__':
    format(sys.stdin)
