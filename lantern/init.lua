lt = {
	F    = require('F'),
	xlua = require('xlua'),

	io   = require('io'),
	os   = require('os'),
	lfs  = require('lfs'),
	term = require('term'),
	path = require('pl.path'),

	nn    = require('nn'),
	optim = require('optim'),
}

local try_require = function(name)
	local success, lib = pcall(function() return require(name) end)
	return {success, lib}
end

local cuda_libs = {
	cutorch = try_require('cutorch'),
	cunn    = try_require('cunn'),
	cudnn   = try_require('cudnn'),
}

for lib, info in pairs(cuda_libs) do
	if info[1] then lt[lib] = info[2] end
end

-- Questionable if we should be doing this by default.
if lt.cudnn then
	lt.cudnn.benchmark = true
	lt.cudnn.fastest = true
	--lt.cudnn.verbose = true
end

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
