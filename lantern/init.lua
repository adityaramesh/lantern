lt = {
	F    = require('F'),
	xlua = require('xlua'),

	io   = require('io'),
	os   = require('os'),
	lfs  = require('lfs'),
	term = require('term'),
	path = require('pl.path'),

	cutorch = require('cutorch'),
	optim   = require('optim'),
}

-- Common utilities.
require('lantern/common')
require('lantern/filesystem')
require('lantern/serializable')
require('lantern/tty_logger')
require('lantern/json_logger')
require('lantern/options')

-- Optimizers.
require('lantern/schedule')
require('lantern/optimizer_base')
require('lantern/adam')
require('lantern/sgu')

-- Core utilities.
require('lantern/state')
require('lantern/checkpointer')
require('lantern/model_base')

return lt
