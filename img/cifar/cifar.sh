#!/bin/sh
wget http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz
python3 cifar2pic.py
