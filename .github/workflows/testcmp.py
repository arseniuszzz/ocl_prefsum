import sys, os
import numpy as np
from datetime import datetime

def readfile(fname, input_size_file):
    print(f"TEST : {datetime.now().strftime("%H:%M:%S")} : {fname} loading...")
    sz = int(open(input_size_file, 'r').readline())
    f = open(fname, 'r')
    data = np.array([0]*sz, dtype=float)
    
    fdata = f.read().rstrip().split(' ')
    print(data.shape, sz, len(fdata))
    for i, y in enumerate(fdata):
        data[i] = float(y)
    return [data, sz]

def testcmp(out, ref, size_file):
    delta = 1e-4
    out_res_l = open(out, 'r').read()
    ref_res_l = open(ref, 'r').read()
    if out_res_l.find("nan") != -1:
        print(f"ERROR : nan in results")			
        return 1
    else:
        out_res, out_size = readfile(out, size_file)
        ref_res, ref_size = readfile(ref, size_file)        
        
        if out_size != ref_size:
            print(f"ERROR : out_size {out_size} != ref_size {ref_size}")			
            return 1
        print(f"TEST : {datetime.now().strftime("%H:%M:%S")} : per-value checking...")
        idx = 0
        while idx < ref_size and (abs(out_res[idx] - ref_res[idx]) / abs(ref_res[idx] + 1e-9)) < delta:
            # print(f"DBG : out_res[{idx}] = {out_res[idx]}, ref_res[{idx}] = {ref_res[idx]}; {abs((out_res[idx] - ref_res[idx]) / (ref_res[idx] + 1e-9))}")		
            idx += 1               
        
        if idx != ref_size:
            print(f"ERROR : {datetime.now().strftime("%H:%M:%S")} : out_res[{idx}] = {out_res[idx]}, ref_res[{idx}] = {ref_res[idx]} [{abs((out_res[idx] - ref_res[idx]) / (ref_res[idx] + 1e-9))}]")			
            return 1
        
        print(f"TEST : {datetime.now().strftime("%H:%M:%S")} : finish checking...")
    return 0


if __name__ == "__main__":
  ref_file = sys.argv[3]
  input_file = sys.argv[1]
  output_file = sys.argv[2]
  exit(testcmp(output_file, ref_file, input_file))
