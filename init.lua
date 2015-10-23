lantern = {}
lantern.class = require "class"

function lantern.make_class(name, base)
	assert(
		not lantern[name],
		"Class `" .. name .. "` already defined."
	)

	local class
	if base then
		class = lantern.class(name, base)
	else
		class = lantern.class(name)
	end

	lantern[name] = class
	return class
end

function lantern.make_accumulator(name)
	assert(
		not lantern.accumulators[name],
		"Accumulator `" .. name .. "` already defined."
	)

	local class = lantern.class(name)
	lantern.accumulators[name] = class
	return class
end

function lantern.make_optimizer(name)
	assert(
		not lantern.optimizers[name],
		"Optimizer `" .. name .. "` already defined."
	)

	local class = lantern.class(name)
	lantern.optimizers[name] = class
	return class
end

lantern.accumulators                    = {}
lantern.performance_metrics             = {}
lantern.performance_metrics["accuracy"] = "increasing"

require "lantern/core/history"

require "lantern/accumulators/accuracy"
require "lantern/accumulators/gradient_norm"
require "lantern/accumulators/zip"

lantern.criterion = {}

require "lantern/criteria"

lantern.momentum   = {}
lantern.schedule   = {}
lantern.optimizers = {}

require "lantern/optimize/momentum"
require "lantern/optimize/schedule"
require "lantern/optimize/sgu"
require "lantern/optimize/rmsprop"
require "lantern/optimize/adadelta"
require "lantern/optimize/adadelta_lm"

-- Define the abstract resources that can be updated by the model driver. We use
-- this convention so that events that are logged can either be printed to
-- stdout or converted into POST responses.
lantern.resources                           = {}
lantern.resources["/history"]               = true
lantern.resources["/history/train_results"] = true
lantern.resources["/history/test_results"]  = true
lantern.resources["/progress"]              = true
lantern.resources["/console/info"]          = true
lantern.resources["/console/warning"]       = true
lantern.resources["/console/error"]         = true

require "lantern/core/io"
require "lantern/core/access_strategy"
require "lantern/core/batch_sampler"
require "lantern/core/batch_provider"
require "lantern/core/driver"

require "lantern/core/csv_logger"
require "lantern/core/stdout_logger"
require "lantern/core/serializer"

require "lantern/core/options"
require "lantern/core/run"

return latern
