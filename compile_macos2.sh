#!/bin/bash

rm printer.o Faddeeva.o 

g++-13 ./source/Faddeeva.cc -c -o Faddeeva.o

g++-13 -c -o printer.o ./source/printer_function.cpp -O3 -std=c++14 -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -fopenmp

g++-13 -o printer1 printer.o Faddeeva.o -O3 -std=c++14 -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -fopenmp

