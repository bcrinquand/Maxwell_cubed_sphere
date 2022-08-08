#!/bin/bash

h5pfc -g -cpp -fcheck=all -O3 -fbounds-check -mcmodel=large loop_field_lines_xyz.f90 -o loop

./loop 