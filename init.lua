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

require "lantern/accumulators/accuracy"
require "lantern/accumulators/zip"

lantern.criterion = {}
require "lantern/criteria"

-- TODO
--
-- lantern.optimizers = {}
-- require "lantern/momentum"
-- require "lantern/schedule"
-- require "lantern/sgu"
-- require "lantern/rmsprop"
-- require "lantern/adadelta"

-- Define the abstract resources that can be updated by the model driver. We use
-- this convention so that events that are logged can either be printed to
-- stdout or converted into POST responses.
lantern.resources                         = {}
lantern.resources["/state"]               = true
lantern.resources["/state/train_results"] = true
lantern.resources["/state/test_results"]  = true
lantern.resources["/progress"]            = true
lantern.resources["/console/info"]        = true
lantern.resources["/console/warning"]     = true
lantern.resources["/console/error"]       = true

require "lantern/core/io"
require "lantern/core/access_strategy"
require "lantern/core/batch_sampler"
require "lantern/core/batch_provider"

require "lantern/core/stdout_logger"
require "lantern/core/serializer"
-- require "lantern/core/epoch"
require "lantern/core/run"

return latern
