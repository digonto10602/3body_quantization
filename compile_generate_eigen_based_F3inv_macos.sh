#!/bin/bash

rm generate_eigen_based_F3inv.o Faddeeva.o

g++-13 ./source/Faddeeva.cc -c -o Faddeeva.o

g++-13 -c -o generate_eigen_based_F3inv.o ./source/generate_eigen_based_F3inv.cpp -O3 -std=c++14 -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -fopenmp

g++-13 -o eigen_F3inv generate_eigen_based_F3inv.o Faddeeva.o -O3 -std=c++14 -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -fopenmp

