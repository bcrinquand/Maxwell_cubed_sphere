#!/bin/bash

num=$(ls ../data_3d/B1u_* | wc -l)
for ((i=0;i<num;i++)); do
    python ddata_cartesiantransform_Bfield.py ${i}
#    python ddata_cartesiantransform_Efield.py ${i}
done
