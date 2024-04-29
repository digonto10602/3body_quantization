#!/bin/bash

rm printer_omp.o Faddeeva.o

g++-13 ./source/Faddeeva.cc -c -o Faddeeva.o

g++-13 -g -c -o printer_omp.o ./source/printer_F3_omp.cpp -O3 -std=c++14 -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -fopenmp

g++-13 -g -o printer_omp printer_omp.o Faddeeva.o -O3 -std=c++14 -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -fopenmp

