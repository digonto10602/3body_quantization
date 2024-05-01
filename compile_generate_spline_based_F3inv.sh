#!/bin/bash

rm generate_spline_based_F3inv.o Faddeeva.o

g++ ./source/Faddeeva.cc -c -o Faddeeva.o

g++ -c -o generate_spline_based_F3inv.o ./source/generate_spline_based_F3inv.cpp -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

g++ -o spline_F3inv generate_spline_based_F3inv.o Faddeeva.o -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

