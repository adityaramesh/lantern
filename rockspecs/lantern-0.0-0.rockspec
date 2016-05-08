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

	-- IO
	--'protobuf',
	'torch_dataflow',
}

build = {
	type = 'builtin',
	modules = {
		['lt.init']               = 'lantern/init.lua',
		['lt.utility.common']     = 'lantern/utility/common.lua',
		['lt.utility.filesystem'] = 'lantern/utility/filesystem.lua',

		['lt.event.proto']            = 'lantern/event/proto.lua',
		['lt.event.event_logger']     = 'lantern/event/event_logger.lua',
		['lt.event.progress_tracker'] = 'lantern/event/progress_tracker.lua',
		['lt.event.event']            = 'lantern/event/event.lua',
		['lt.event.event_group']      = 'lantern/event/event_group.lua',
		['lt.event.loss']             = 'lantern/event/loss.lua',
	}
}
