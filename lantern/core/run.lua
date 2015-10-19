--
-- This file provides a convenience high-level function called `run`, for
-- training or evaluating a model on a given task. The user can invoke this
-- function to get started with a minimal amount of boilerplate code. Several
-- optional arguments allow a great deal of flexibility and customization of the
-- training process.
--

-- We do careful verification here to minimize our risk of running into
-- easily-preventable problems much later down the line.
local function validate_args(args)
	assert(args)
	assert(type(args) == "table")
	assert(args.model)
	assert(args.driver)
	assert(args.perf_metrics)
	assert(type(args.perf_metrics) == "table")
	assert(args.model_dir)
	assert(type(args.model_dir) == "string")

	-- Convert `perf_metrics` to a key-value table if it is an array.
	if args.perf_metrics[1] then
		local tmp = {}
		for _, v in pairs(args.perf_metrics) do
			-- Here, `true` is just used as a dummy value.
			tmp[v] = true
		end
		args.perf_metrics = tmp
	end

	for k, _ in pairs(args.perf_metrics) do
		assert(
			lantern.performance_metrics[k],
			"Unrecognized metric `" .. k .. "`."
		)
	end
end

--
-- The arguments to this function must be provided in a table.
--
-- Parameters:
-- * model: An implementation of the Model interface.
-- * driver: Class responsible for performing the training and testing epochs.
-- * perf_metrics: A table of strings indicating which performance metrics to
--   compute. If `perf_metrics[m] == "not important"` for some metric `m`, then
--   the model will not necessarily be saved each time `m` is improved. By
--   default, the model is saved if **any** performance metric improves after a
--   training epoch. Valid values: any metric in the table
--   `lantern.performance_metrics`.
-- * model_dir: Directory in which to serialize models and other state
--   information.
-- * history (optional): Used to resume progress from a previous training
--   trajectory.
-- * test_epoch_ratio (optional): The number of training epochs to perform
--   before running a test epoch. Default: 1.
-- * optimizer (optional): Optimizer used to update the model. The default
--   optimizer is `lantern.optimizers.adadelta_lm` with default parameters.
-- * stop_crit (optional): Criterion used to determine when to stop training.
--   Default: stop if there is no improvement in **any** performance metric
--   after 10 epochs.
-- * logger (optional): Used to log real-time events. Default:
--   `lantern.stdout_logger`.
--
function lantern.run(args)
	validate_args(args)

	-- Make aliases to the required arguments.
	local model        = args.model
	local model_dir    = args.model_dir
	local driver       = args.driver
	local perf_metrics = args.perf_metrics

	-- Infer the values of the optional arguments.
	local optim            = args.optimizer or lantern.optimizers.adadelta_lm(model)
	local stop_crit        = args.stop_crit or lantern.criterion.max_epochs_per_improvement(10)
	local test_epoch_ratio = args.test_epoch_ratio or 1
	local logger           = args.logger or lantern.stdout_logger()
	local hist             = args.history or {{epoch = 1}}

	assert(#hist >= 1)
	logger:update("/history", hist)

	local make_accumulator = function()
		local accs = {}
		for k, _ in pairs(perf_metrics) do
			if k == "accuracy" then
				assert(#model:output_shape() == 1)
				accs[#accs + 1] = lantern.accumulators.accuracy(
					model:output_shape()[1])
			else
				error("Unrecognized metric `" .. k .. "`.")
			end
		end
		return lantern.accumulators.zip(accs)
	end

	local time_epoch = function(func)
		local start   = sys.clock()
		local metrics = func()
		metrics.time  = sys.clock() - start
		return metrics
	end

	local train_func = function()
		return driver:train_epoch(model, optim, make_accumulator(), logger)
	end

	local test_func = function()
		return driver:test_epoch(model, optim, make_accumulator(), logger)
	end

	if driver.train then
		if (hist[#hist].train or hist[#hist].test) and not stop_crit(hist) then
			logger:update("/console/info", "Stopping criterion satisfied.")
			return
		elseif hist[#hist].train and hist[#hist].test then
			local cur_epoch = #hist
			hist[cur_epoch + 1] = {epoch = cur_epoch + 1}
		end

		local serializer = lantern.serializer(model_dir, perf_metrics, logger)

		while true do
			local cur_entry = hist[#hist]
			local cur_epoch = cur_entry.epoch
			assert(cur_epoch > 0)

			logger:update("/console/info", "Starting epoch " .. cur_epoch .. ".")

			if not cur_entry.train then
				logger:update("/console/info", "Starting training epoch.")
				local data = time_epoch(train_func)
				cur_entry["train"] = data
				logger:update("/history/train_results", data)
				serializer:save_train_progress(model, optim, hist, logger)
			end

			if not cur_entry.test and (cur_epoch - 1) % test_epoch_ratio == 0 then
				logger:update("/console/info", "Starting testing epoch.")
				local data = time_epoch(test_func)
				cur_entry["test"] = data
				logger:update("/history/test_results", data)
				serializer:save_test_progress(model, optim, hist, logger)
			end

			if not stop_crit(hist) then
				logger:update("/console/info", "Stopping criterion satisfied.")
				return
			end

			hist[cur_epoch + 1] = {epoch = cur_epoch + 1}
		end
	else
		logger:update("/console/info", "Starting testing epoch.")
		local data = time_epoch(test_func)
		logger:update("/history/test_results", data)
	end
end
