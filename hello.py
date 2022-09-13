#!/usr/bin/env python
import math
import maestro
import os


#def w1(str):
#    print (str)
#    wait = input()
#    return
#
#print(os.getpid())
#w1('starting main..press a key')
try:
    var = maestro.main_m(4096)
except Exception as e:
    print(type(e))
    print(e)
    var = -1


def is_float(val):
    try:
        float(val)
    except ValueError:
        return False
    else:
        return True
def is_nan(val):
    try:
        x = float(val)
    except ValueError:
        return True

    if(math.isnan(x)):
        return True
    else:
        return False

#print(var)
#print("[ 0:runtime, 1:engergy, 2:throughtput, 3:computation, 4:l1_size, 5:l2_size, 6:area, 7:power, 8:ddr_energy, 9:num_pe_utilized, 10:reuse_input, 11:reuse_weight, 12:reuse_output]")
name_list = [ "runtime", "engergy", "throughtput", "computation", "l1_size", "l2_size", "area", "power", "ddr_energy", "num_pe_utilized", "reuse_input", "reuse_weight", "reuse_output", "l2_regy", "l2_wegy", "l1_regy", "l1_wegy", "mac_egy"]

print("[",end='')
i = 0
for v in var:
    tmp = str(int(v))
    print(name_list[i]+":"+tmp+", ",end='')
    i = i+1

print("]",end='')
print("")