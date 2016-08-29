package = 'lantern'
version = '0.0-1'

source = {
	-- Use this for quick installation from local sources during development.
	url = '.',
	dir = '.',
	-- url = 'git://github.com/adityaramesh/lantern',
	-- branch = 'master'
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
		['lt.init']           = 'lantern/init.lua',
		['lt.common']         = 'lantern/common.lua',
		['lt.filesystem']     = 'lantern/filesystem.lua',
		['lt.serializable']   = 'lantern/serializable.lua',
		['lt.tty_logger']     = 'lantern/tty_logger.lua',
		['lt.json_logger']    = 'lantern/json_logger.lua',
		['lt.options']        = 'lantern/options.lua',

		['lt.schedule']       = 'lantern/schedule.lua',
		['lt.optimizer_base'] = 'lantern/optimizer_base.lua',
		['lt.adam']           = 'lantern/adam.lua',
		['lt.sgu']            = 'lantern/sgu.lua',

		['lt.global_rng_state'] = 'lantern/global_rng_state.lua',
		['lt.checkpointer']     = 'lantern/checkpointer.lua',
		['lt.model_base']       = 'lantern/model_base.lua',
	}
}
