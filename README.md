# Overview

Lantern provides utilities for writing maintainable, robust, and efficient, training scripts in
Torch. Here is an abbreviated list of features:

- [Predefined command-line options](lantern/options.lua) to make it easy to start and resume
  experiments.
- [A checkpointer](lantern/checkpointer.lua) that maintains the states of a list of specified
  objects on the disk.
- [A base class for models](lantern/model_base.lua) that automatically frees unnecessary storage and
  transfers parameters to the CPU when the model is written to disk.
- [A JSON logger](lantern/json_logger.lua) to allow for easy monitoring and analysis of data from
  experiments.

Starting a project using Lantern typically involves three steps. We have provided a working example
of an MNIST classifier in the `examples` directory for your reference. [This section](#Running the
Example) describes how to get it running.

1. Define the model (see [this file](examples/model.lua) for reference).
2. Write a driver script to train the model (see [this file](examples/driver.lua) for reference; if
   you are training a classifier, this example script may already be sufficient for your purposes).
3. Write a "front-end" script to be invoked from the command-line (see [this
   file](examples/train_classifier.lua) for reference). 

The model, driver, and front-end scripts are usually independent components of any given experiment.
Separating these components allows different versions of each to be developed over time, without the
maintainability of the project being compromised by many slightly different versions of the same
code.

# Installation

	luarocks install https://raw.githubusercontent.com/adityaramesh/lantern/master/rockspecs/lantern-0.0-1.rockspec

# Running the Example

First, you will need to download the example code and preprocessed data:

	git clone https://github.com/adityaramesh/lantern
	cd lantern
	chmod +x examples/download_data.sh
	./examples/download_data.sh

Now you can begin training the classifier as follows:

	mkdir experiments
	th example/train_classifier.lua -task create -experiment test -gpu 1

Henceforth, attempting to rerun the training script using the same options will fail. Instead, you
have two options:

- Use `-task resume`, which will resume training from the start of the first unfinished epoch.
- Use `-task replace`, which will **purge all data** from previous runs and train a new model from
  scratch.

You can change the relative path to the root directory for experiments using the option
`-experiment_root`, which defaults to `experiments`.
