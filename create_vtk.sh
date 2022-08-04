#!/bin/bash

files=(./data/Bsq_*.h5)
do fil in files
    h5tovtk ${fil}
done