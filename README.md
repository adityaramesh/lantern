# Overview

Lantern makes it easy to train models in Torch and monitor real-time performance
statistics without having to write any boilerplate code.

# Agenda

- Change to the same package system in `torch_dataflow`.
- Use single quotes where appropriate.
- Change line width to 100.
- Minimize use of global definitions in `init.lua`.

# Future Features

- Support for logging values at multiple levels of temporal granularity (e.g. every batch, every k
  batches, every epoch, etc.). For large datasets, the storage requirements for this data can
  quickly grow large, so we need to be careful about how we store and serialize it. A good solution
  will not fit into the existing serialization framework.
  - The right way to do this is probably to use something like TensorFlow's `SummaryWriter`:
    https://www.tensorflow.org/versions/r0.7/how_tos/summaries_and_tensorboard/index.html. This
    allows one to log many events during training, without slowing down the driver thread with IO
    operations.
  - To implement this concept Torch, it may be best to implement an eventfd-based writer in C++, and
    interface to it from Lua.
  - To visualize these high-frequency events, it makes the most sense to rely on the browser and use
    JS, like TensorBoard.

- Stopping criterion for max time (by extrapolating how many epochs we can do based on the time
  taken by previous ones).

- Integration with neo, for fast IO.
- Integration with raze, for distributed optimization using multiple GPUs.
