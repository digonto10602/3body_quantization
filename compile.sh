#!/bin/bash


g++ Faddeeva.cc -c -o Faddeeva.o

g++ -c -o printer.o printer_function.cpp -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

g++ -o printer printer.o Faddeeva.o -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

