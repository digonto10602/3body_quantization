#!/bin/bash

rm generate_K3iso.o Faddeeva.o 

g++-13 ./source/Faddeeva.cc -c -o Faddeeva.o

g++-13 -c -o generate_K3iso.o ./source/generate_K3iso.cpp -O3 -std=c++14 -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -fopenmp

g++-13 -o generate_K3iso generate_K3iso.o Faddeeva.o -O3 -std=c++14 -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -fopenmp

