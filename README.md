# Overview

Lantern makes it easy to train models in Torch and monitor real-time performance
statistics without having to write any boilerplate code.

# Future Features

- "Ready-to-go" models in the namespace `lantern.models`. The layers sizes and
  depth are automatically chosen based on the size of the input. Probably
  better to put these in a different repository.
  - CNN with 5x5 filters.
  - VGG-style net with 3x3 filters. (Make it easy to do the successive seeding
    operations to construct deeper VGG nets.)
  - NIN, Maxout, PReLU.
  - "Inception" modules for GoogLeNet.

- Stopping criterion for max time (by extrapolating how many epochs we can do
  based on the time taken by previous ones).

- Generalize the "train_files" parameter to a list of abstract resources that
  can either be file names or classes. This allows the batch provider to
  interact with classes that generate data on the fly.

- Logger for unix domain sockets instead of stdout.
