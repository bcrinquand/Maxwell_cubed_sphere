#!/bin/bash

num=116 ##$(ls ../data_3d_gr/B1u_* | wc -l)
for ((i=65;i<num;i++)); do
    python gr_ddata_cartesiantransform_Bfield_int_half.py ${i}
    python gr_ddata_cartesiantransform_Dfield_int_half.py ${i}
done
