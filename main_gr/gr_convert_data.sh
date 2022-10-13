#!/bin/bash

num=199 ##$(ls ../data_3d_gr/B1u_* | wc -l)
for ((i=0;i<num;i++)); do
    python gr_ddata_cartesiantransform_Bfield.py ${i}
    python gr_ddata_cartesiantransform_Dfield.py ${i}
done
