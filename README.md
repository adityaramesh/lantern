# Overview

Lantern makes it easy to train models in Torch and monitor real-time performance
statistics without having to write any boilerplate code.

# Agenda

## Stage 2
- File with default training and testing function implementations
- Port adadelta
- Port the csv logger so that we can use it for the optimizers
- Function to set up default command-line arguments
- Test on a simple example

## Stage 3
- Port the other optimizers

# Future Features

- Stopping criterion for max time (by extrapolating how many epochs we can do
  based on the time taken by previous ones)

- Generalize the "train_files" parameter to a list of abstract resources that
  can either be file names or classes. This allows the batch provider to
  interact with classes that generate data on the fly.

- Logger for unix domain sockets instead of stdout.
