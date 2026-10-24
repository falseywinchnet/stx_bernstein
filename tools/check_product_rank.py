"""Exact modular rank certificates for sampled pair-product columns."""
import json
from math import comb
from pathlib import Path

def rank(rows, prime):
    basis = {}
    pivots = []
    for index, original in enumerate(rows):
        row = [v % prime for v in original]
        for column in range(len(row)):
            if row[column] == 0:
                continue
            if column in basis:
                factor = row[column]
                other = basis[column]
                for j in range(column,len(row)):
                    row[j] = (row[j]-factor*other[j])%prime
            else:
                inv = pow(row[column],prime-2,prime)
                basis[column] = [(v*inv)%prime for v in row]
                pivots.append(index)
                break
        if len(basis) == len(rows[0]):
            break
    return len(basis), pivots

def axis(length,quantized):
    extent=max(length-1,1)
    result=[]
    for position in range(length):
        if quantized:
            current=[1048576]+[0]*8
            for degree in range(1,9):
                following=[0]*9
                for i in range(degree):
                    right=current[i]*position//extent
                    following[i]+=current[i]-right
                    following[i+1]+=right
                current=following
        else:
            # Common denominator extent**8 can be omitted without changing rank.
            current=[comb(8,i)*position**i*(extent-position)**(8-i) for i in range(9)]
        result.append([current[i]*current[j] for i in range(9) for j in range(i,9)])
    return result

results=[]
for length in [17,31,64,193,512,1024]:
    for quantized in [False,True]:
        matrix=axis(length,quantized)
        certificates=[]
        for prime in [2147483647,1000000007]:
            dimension,pivots=rank(matrix,prime)
            certificates.append({'prime':prime,'rank_mod_prime':dimension,'pivot_rows':pivots})
        results.append({'length':length,'basis':'Q20 recurrence' if quantized else 'exact Bernstein','certificates':certificates})
        print(length,results[-1]['basis'],[c['rank_mod_prime'] for c in certificates],flush=True)
Path(__file__).with_name('results.json').write_text(json.dumps(results,indent=2)+'\n')
