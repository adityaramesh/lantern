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
	modules = {lantern = "init.lua"},
	copy_directories = {"lantern", "lantern/accumulators", "lantern/core", "lantern/optimize"}
}
