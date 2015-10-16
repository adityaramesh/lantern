require "cutorch"
local class = require "class"

dofile "init.lua"
local dummy_model = class("dummy_model")

function dummy_model:__init()
	self.params      = torch.CudaTensor(10):zero()
	self.grad_params = torch.CudaTensor(10)
	self.outputs     = torch.CudaTensor{1, 0, 0, 0, 0}
end

function dummy_model:parameters()
	return self.params
end

function dummy_model:grad_parameters()
	return self.grad_params
end

function dummy_model:input_shape()
	return torch.LongStorage{32, 32}
end

function dummy_model:output_shape()
	return torch.LongStorage{10}
end

function dummy_model:forward(inputs, targets)
	if inputs:nDimension() == 2 then
		return self.outputs, 100
	elseif inputs:nDimension() == 3 then
		local bs = inputs:size(1)
		return torch.repeatTensor(self.outputs, bs, 1), bs * 100
	else
		error("Unexpected number of input dimensions.")
	end
end

function dummy_model:evaluate(inputs, targets)
	return self:forward(inputs, targets)
end

local info = lantern.parse_options()

local bp = lantern.batch_provider({
	train_files = {
		"data/mnist/partitioned/train_left.t7",
		"data/mnist/partitioned/train_right.t7"
	},
	test_file  = "data/mnist/scaled/test_32x32.t7",
	target     = "gpu",
	batch_size = 200
})

lantern.run({
	model        = dummy_model(),
	driver       = lantern.driver(bp),
	perf_metrics = {"accuracy"},
	model_dir    = info.model_dir,
	optimizer    = info.optimizer,
	history      = info.history
})
