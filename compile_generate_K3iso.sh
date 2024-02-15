#!/bin/bash

rm generate_K3iso.o Faddeeva.o 

g++ ./source/Faddeeva.cc -c -o Faddeeva.o

g++ -c -o generate_K3iso.o ./source/generate_K3iso.cpp -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

g++ -o generate_K3iso generate_K3iso.o Faddeeva.o -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

