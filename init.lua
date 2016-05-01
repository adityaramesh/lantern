lantern = {

}

lantern.accumulators                    = {}
lantern.performance_metrics             = {}
lantern.performance_metrics["accuracy"] = "increasing"

require "lantern/core/history"

-- DONE
require "lantern/accumulator/accuracy"
require "lantern/accumulator/gradient_norm"
require "lantern/accumulator/zip"

lantern.criterion = {}

-- DONE
require "lantern/core/criterion"

lantern.momentum   = {}
lantern.schedule   = {}
lantern.optimizers = {}

-- DONE
require "lantern/optimizer/momentum"
require "lantern/optimizer/schedule"
require "lantern/optimizer/sgu"
require "lantern/optimizer/rmsprop"
require "lantern/optimizer/adadelta"
require "lantern/optimizer/adadelta_lm"

-- Define the abstract resources that can be updated by the model driver. We use this convention so
-- that events that are logged can either be printed to stdout or converted into POST responses.
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
