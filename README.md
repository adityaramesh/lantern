# Overview

Lantern makes it easy to train models in Torch and monitor real-time performance statistics without
having to write any boilerplate code.

# Agenda

Put this in the driver:

	--[[
	All classes in lantern use local RNGs instead of the global ones, but we seed the global
	RNGs anyway, so that third-party functions that rely on random-number generation (including
	those in Torch) yield reproducible results.
	--]]
	torch.manualSeed(1)
	cutorch.manualSeed(1)

- [x] Revised default options.
  - Things to include: task, experiment, version ('current' or best associated with a particular
    metric), `experiment_root` (default: `./experiments`).
  - We use separate scripts for training and evaluation, so the only options for `task` should be
    `create`, `resume`, and `replace`.

- [ ] Event API.
  - [ ] `loss.lua`
  - [ ] `progress_tracker.lua` and test
  - [ ] `confusion.lua`
    - Visualize the class-conditional distribution using a donut.
  - [ ] `accuracy.lua`
  - [ ] `balanced_accuracy.lua` and test for both accuracy events
    - Weighting by inverse frequency (`f_k := t_k / N`) is equivalent to weighting by `1 / t_k`,
      which reduces to regular accuracy in the case where all classes have the same number of
      instances.

- [ ] Checkpoint class.
  - Put this in `core` or `event`?
  - `<experiment_name>/current/{files}` or
    `<experiment_name>/<dataset_name>/best_<event_name>/{files}`, where `{files}` consists of:
      - `event_data.dat` (measurements from each event group; messagepack format)
        - This file also contains non-numeric data like images and audio in binary format. They are
	  not written to files in separate directories, because it makes it much easier and more
	  efficient to checkpoint event data when it is all in a single file.
      - `best_metrics.dat` (best metrics as saved by the progress tracker).
  - Test

- [ ] Consolidate `run.lua` and `driver.lua`.
  - Accept the paths from `parse_options` and perform the deserialization/construction from scratch
    manually rather than making the user check.
  - Revise the model interface so that `forward` and `backward` are used instead of `predict` and
    `evaluate`. Create a separate abstract model class to help with this.
      - Also have the model register events with the event group provided for each data set, rather
        than registering the events in the script that is run. This makes more sense and keeps the
	code modular, since model events are dealt with in the `forward` function anyway.
  - Make the default driver an instance of the `call` flow graph agent.
  - The data sources are already provided, since they will be the parents of the driver node in the
    flow graph.
  - Default options: model, optimizer, stopping criterion, logger. A table containing information
    about each parent (data source) in the flow graph must also be provided. This information
    includes:
    - Epoch period (i.e. how many times the epoch counter must be advanced to perform an epoch on
      this data source). The default is 1.
    - Event group (default is an empty event group, so that it can still be used by the driver and
      optimizer to register default events).
  - Verbose option for driver; this causes it to register additional events for timing IO,
    fprop/bprop, etc., with each provided event group.
      - Events added by the library should have names beginning with an underscore, so that no name
        clashes occur with user-defined names.

- [ ] Integrate optimizers with the serialization and event API.
  - Include a "verbose" option for events as with the driver.

- [ ] Revise stopping criteria.
- [ ] Revise stdout logger so that it logs to a general file instead, with one option being stdout.
  Perhaps make this another default option, and create the logger automatically?

# Future Features

- Default features to have:
  - Automatic timing of IO and model evaluation time; warning if IO time is greater than model
    evaluation time, in which case the IO will no longer be overlapped.
  - Option to automatically save images of a fixed set of filters for each layer of the network
    every once in a while.
  - Utility for computing statistics about filter saturation.
  - Utility for computing statistics about the gradient from each module.
  - Bokeh plot with live updates every few seconds, hooked up to the CSV file being written to.
  - Slider widget for iPython that can be used to scan through a directory of images, e.g.
    `foo_1.png`, `foo_2.png`, etc. This is useful for visualizing filters and generated images.
  - Automatic logging of confusion matrix; automatic computation of statistics for most confusable
    classes, most difficult class to correctly classify, easiest classes to classify, etc.

- Stopping criterion for max time (by extrapolating how many epochs we can do based on the time
  taken by previous ones).

- Integration with raze, for fast IO and distributed optimization using multiple GPUs.
