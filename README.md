# Overview

Lantern makes it easy to train models in Torch and monitor real-time performance
statistics without having to write any boilerplate code.

# Agenda

- Support for multiple validation and test sets (so that we can do things like keep separate models
  for noisy speech and clean speech). The "test epoch" should actually function as a "meta test
  epoch", in which we perform one pass over all of the validation sets that are currently
  registered.
- By default, an accumulator should attach to all data sets; the user should be able to optionally
  provide the names of the data sets to which the accumulator should be attached, in case the
  default behavior is not desired.
- The `make_accumlator` function in `run.lua` needs to be refatored and revised to allow for all of
  this.
- Integrate balanced class sampling with framework. How best to do this?
- Stopping criterion for max time (by extrapolating how many epochs we can do based on the time
  taken by previous ones).

# Future Features

- Generalize the "train_files" parameter to a list of abstract resources that can either be file
  names or classes. This allows the batch provider to interact with classes that generate data on
  the fly.
