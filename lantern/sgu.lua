local sgu, parent = torch.class('lt.sgu', 'lt.optimizer_base')
local momentum_types = {none = true, classic = true, nag = true}

--[[
Required parameters:
* `tensor_type`
* `tensor_size`

Optional Parameters:
* `step_size`: Default is 1e-3.
* `momentum`: Default is 0.95.
* `momentum_type`: Default is 'nag'. Options: 'none', 'classic', or 'nag'.
* `logger`
--]]
function sgu:__init(args)
	assert(args.momentum_type == nil or momentum_types[args.momentum_type])
	parent.__init(self, args)

	self.state.iter          = 0
	self.state.first         = true
	self.state.step_size     = args.step_size     or lt.constant(1e-3)
	self.state.momentum      = args.momentum      or lt.constant(0.95)
	self.state.momentum_type = args.momentum_type or 'nag'

	if type(self.state.step_size) == 'number' then
		self.state.step_size = lt.constant(self.state.step_size)
	end

	if type(self.state.momentum) == 'number' then
		self.state.momentum = lt.constant(self.state.momentum)
	end

	self:initialize()
end

function sgu:initialize()
	parent.initialize(self)

	if self.state.momentum_type == 'nag' then
		if self.state.step == nil then
			self.state.step = self.tensor_factory():resize(self.state.tensor_size)
		elseif self.state.step:type() ~= self.state.tensor_type then
			-- This branch is taken during deserialization.
			self.state.step = self.state.step:type(self.state.tensor_type)
		end
	end
end

function sgu:update(params, eval_func)
	self.state.iter = self.state.iter + 1
	local cur_step_size = self.state.step_size(self.state.iter)
	assert(cur_step_size > 0 and cur_step_size <= 1)

	if self.state.momentum_type == 'none' then
		local _, grad_params = eval_func(params)
		params:add(-cur_step_size, grad_params)
	elseif self.state.momentum_type == 'nag' then
		if self.state.first then
			local _, grad_params = eval_func(params)
			self.state.step:copy(grad_params):mul(-cur_step_size)
			params:add(self.state.step)
			self.state.first = false
			return
		end

		local cur_mom = self.state.momentum(self.state.iter)
		assert(cur_mom > 0 and cur_mom < 1)

		-- Evaluate the function at the trial point.
		self.state.step:mul(cur_mom)
		params:add(self.state.step)
		local _, grad_params = eval_func(params)

		self.state.step:add(-cur_step_size, grad_params)
		params:add(-cur_step_size, grad_params)
	else
		error(F"Unsupported momentum type '{self.state.momentum_type}'.")
	end
end

function sgu:__save(args)
	local temp_step = self.state.step

	if self.state.momentum_type == 'nag' and self.state.tensor_type == 'torch.CudaTensor' then
		self.state.step = self.state.step:float()
	end

	parent.__save(self, args)

	if self.state.momentum_type == 'nag' and self.state.tensor_type == 'torch.CudaTensor' then
		self.state.step = temp_step
	end

	collectgarbage()
end
