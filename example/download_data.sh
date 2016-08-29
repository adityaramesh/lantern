#! /usr/bin/env bash

mkdir -p data/mnist/padded
cd data/mnist/padded

if [[! -f train.hdf5 ]]; then
	wget https://github.com/adityaramesh/lantern/releases/download/data/train.hdf5
fi

if [[! -f test.hdf5 ]]; then
	wget https://github.com/adityaramesh/lantern/releases/download/data/test.hdf5
fi
