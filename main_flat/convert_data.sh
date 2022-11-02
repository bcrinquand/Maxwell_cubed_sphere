#!/bin/bash

num=136 #$(ls ../data_3d/B1u_* | wc -l)
for ((i=99;i<num;i++)); do
    python ddata_cartesiantransform_Bfield.py ${i}
    python ddata_cartesiantransform_Efield.py ${i}
done
