#!/bin/bash

rm generate_pole.o Faddeeva.o 

g++ ./source/Faddeeva.cc -c -o Faddeeva.o

g++ -c -o generate_pole.o ./source/generate_pole.cpp -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

g++ -o generate_pole generate_pole.o Faddeeva.o -O3 -std=c++14 -I/usr/include/eigen3/ -fopenmp

