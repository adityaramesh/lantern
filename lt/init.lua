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
require('lt/common')
require('lt/filesystem')
require('lt/serializable')
require('lt/tty_logger')
require('lt/json_logger')
require('lt/options')

-- Optimizers.
require('lt/schedule')
require('lt/optimizer_base')
require('lt/adam')
require('lt/sgu')

-- Core utilities.
require('lt/global_rng_state')
require('lt/checkpointer')
require('lt/model_base')

return lt
