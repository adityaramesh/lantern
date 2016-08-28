local cutorch = lt.cutorch
local model_base, parent = torch.class('lt.model_base', 'lt.serializable')

function model_base:__init()
	parent.__init(self)
end

function model_base:initialize()
	if cutorch then self.state.module:cuda() end
	self.params, self.grad_params = self.state.module:getParameters()
end

function model_base:output()
	return self.state.module.output
end

function model_base:parameters()
	return self.params
end

function model_base:grad_input()
	return self.state.module.gradInput
end

function model_base:grad_parameters()
	return self.grad_params
end

function model_base:training()
	self.state.module:training()
end

function model_base:evaluate()
	self.state.module:evaluate()
end

function model_base:forward(input)
	return self.state.module:forward(input)
end

function model_base:backward(input, grad_output)
	return self.state.module:backward(input, grad_output)
end

function model_base:update_grad_input(input, grad_output)
	return self.state.module:updateGradInput(input, grad_output)
end

function model_base:acc_grad_parameters(input, grad_output, scale)
	return self.state.module:accGradParameters(input, grad_output, scale)
end

--[[
Provides a standard API for derived classes to implement behavior like zeroing out biases of
convolutional layers after each update. The driver is expected to invoke this function on a model
after the optimizer is applied to it.
]]--
function model_base:notify_post_update() end

function model_base:apply(f)
	return self.state.module:apply(f)
end

function model_base:replace(f)
	return self.state.module:replace(f)
end

function model_base:__write(file)
	self.state.module:clearState()
	local copy = self.state.module:clone()
	copy:float()

	local temp = self.state.module
	self.state.module = copy
	parent.__write(self, file)

	self.state.module = temp
	copy = nil
	collectgarbage()
end

function model_base:__read(file)
	parent.__read(self, file)
	self:initialize()
end
