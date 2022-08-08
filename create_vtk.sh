#!/bin/bash

for fil in ./data/Bsq_*.h5; do
    echo ${fil}
    h5tovtk ${fil}
done