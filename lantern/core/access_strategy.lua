require "torch"
local shuffled_access_strategy = lantern.make_class("shuffled_access_strategy")
local linear_access_strategy   = lantern.make_class("linear_access_strategy")

local function validate_data(data)
	assert(type(data) == "table")
	assert(#data >= 1)
end

function shuffled_access_strategy:__init(data)
	validate_data(data)
	self.data = data

	self.perms = {}
	for i = 1, #data do
		self.perms[i] = torch.randperm(self.data[i].inputs:size(1))
	end
end

function shuffled_access_strategy:index(dataset, offset)
	assert(dataset >= 1)
	assert(dataset <= #self.data)
	assert(offset >= 1)
	assert(offset <= self.data[dataset].inputs:size(1))

	return self.perms[dataset][offset]
end

function linear_access_strategy:__init(data)
	validate_data(data)
end

function linear_access_strategy:index(dataset, offset)
	return offset
end
