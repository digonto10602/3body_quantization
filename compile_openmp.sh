#!/bin/bash

rm printer_omp.o Faddeeva.o 

g++ ./source/Faddeeva.cc -c -o Faddeeva.o -fopenmp 

g++ -g -c -o printer_omp.o ./source/printer_F3_omp.cpp -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp 

g++ -g -o printer_omp printer_omp.o Faddeeva.o -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp 

