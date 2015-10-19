package = "lantern"
version = "0.1-1"

source = {
	url = "git://github.com/adityaramesh/lantern",
	tag = "master",
	dir = "lantern"
}

description = {
	summary  = "Train models and monitor progress without boilerplate.",
	homepage = "https://github.com/adityaramesh/lantern",
	license  = "BSD 3-Clause"
}

dependencies = {
	"class >= 0.5.0",
	"torch >= 7.0",
	"xlua >= 1.0",
	"hdf5 >= 0.0",
	"lunajson >= 1.1"
}

build = {
	type = "builtin",
	modules = {
		lantern = {
			"lantern/criteria.lua",
			"lantern/accumulators/accuracy.lua",
			"lantern/accumulators/zip.lua",
			"lantern/core/access_strategy.lua",
			"lantern/core/batch_provider.lua",
			"lantern/core/batch_sampler.lua",
			"lantern/core/csv_logger.lua",
			"lantern/core/driver.lua",
			"lantern/core/io.lua",
			"lantern/core/options.lua",
			"lantern/core/run.lua",
			"lantern/core/serializer.lua",
			"lantern/core/stdout_logger.lua",
			"lantern/optimize/adadelta.lua",
			"lantern/optimize/adadelta_lm",
			"lantern/optimize/momentum.lua",
			"lantern/optimize/rmsprop.lua",
			"lantern/optimize/schedule.lua",
			"lantern/optimize/sgu.lua",
			"init.lua"
		}
	}
}
