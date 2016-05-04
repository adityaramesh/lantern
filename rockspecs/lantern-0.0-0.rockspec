package = 'lantern'
version = '0.0-0'

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
	license  = 'BSD 3-Clause'
}

dependencies = {
	'torch >= 7.0',

	-- Utilities
	'fun',
	'f-strings',
	'totem',

	-- OS
	'luafilesystem',
	'luaposix',
}

build = {
	type = 'builtin',
	modules = {
		['lt.init']        = 'lantern/init.lua',
		['lt.event.proto'] = 'lantern/event/proto.lua',
	}
}
