#!/bin/bash

rm printer_acc.o Faddeeva.o 

nvc++ ./source/Faddeeva.cc -c -o Faddeeva.o -acc=host

nvc++ -c -o printer_acc.o ./source/printer_F3_acc.cpp -O3 -std=c++14 -I/usr/include/eigen3/ -acc=host 

nvc++ -o printer_acc printer_acc.o Faddeeva.o -O3 -std=c++14 -I/usr/include/eigen3/ -acc=host

