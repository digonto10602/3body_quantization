#!/bin/bash

rm generate_K3iso.o Faddeeva.o 

g++ ./source/Faddeeva.cc -c -o Faddeeva.o

g++ -c -o generate_F3iso.o ./source/generate_F3iso.cpp -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

g++ -o generate_F3iso generate_F3iso.o Faddeeva.o -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

