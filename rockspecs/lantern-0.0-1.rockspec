package = 'lantern'
version = '0.0-1'

source = {
	-- Use this for quick installation from local sources during development.
	-- url = '.',
	-- dir = '.',
	url = 'git://github.com/adityaramesh/lantern',
	branch = 'master'
}

description = {
	summary = "A model training and performance monitoring framework for Torch.",
	homepage = 'https://github.com/adityaramesh/lantern',
	license  = 'BSD 3-Clause',
}

dependencies = {
	'torch >= 7.0',
	'optim',

	'penlight',
	'f-strings',

	'luafilesystem',
	'luaposix',
	'lua-term',
}

build = {
	type = 'builtin',
	modules = {
		['lt.init']           = 'lt/init.lua',
		['lt.common']         = 'lt/common.lua',
		['lt.filesystem']     = 'lt/filesystem.lua',
		['lt.serializable']   = 'lt/serializable.lua',
		['lt.tty_logger']     = 'lt/tty_logger.lua',
		['lt.json_logger']    = 'lt/json_logger.lua',
		['lt.options']        = 'lt/options.lua',

		['lt.schedule']       = 'lt/schedule.lua',
		['lt.optimizer_base'] = 'lt/optimizer_base.lua',
		['lt.adam']           = 'lt/adam.lua',
		['lt.sgu']            = 'lt/sgu.lua',

		['lt.global_rng_state'] = 'lt/global_rng_state.lua',
		['lt.checkpointer']     = 'lt/checkpointer.lua',
		['lt.model_base']       = 'lt/model_base.lua',
	}
}
