lantern = {
	--
	-- Packages used internally are defined in advance to avoid redundancy.
	--
	F = require('F'),
	fun = require('fun'),

	os = require('os'),
	path = require('pl.path'),

	--
	-- Definitions global to lantern.
	--
	accumulator = {},
	criterion = {},

	momentum = {},
	schedule = {},
	optimizer = {},
}

local F = lantern.F

--[[
Wrapper around Torch's `class` function with extra checks to ensure that there are no name
conflicts.
--]]
function lantern.make_class(name, base)
	assert(lantern[name] == nil, F"Class '{name}' is already defined.")

	if base == nil then
		torch.class(name, lantern)
	else
		torch.class(name, base, lantern)
	end

	return lantern[name]
end

function lantern.make_accumulator(name)
	assert(not lantern.accumulator[name], F"Accumulator '{name}' already defined.")

	local class = lantern.class(name)
	lantern.accumulator[name] = class
	return class
end

function lantern.make_optimizer(name)
	assert(not lantern.optimizer[name], F"Optimizer '{name}' already defined.")

	local class = lantern.class(name)
	lantern.optimizer[name] = class
	return class
end

-- torch.include(...)

torch.include('lantern/optimizer/momentum')
torch.include('lantern/optimizer/schedule')

torch.include('lantern/optimizer/sgu')
torch.include('lantern/optimizer/rmsprop')
torch.include('lantern/optimizer/adadelta')
torch.include('lantern/optimizer/adadelta_lm')

return lantern
