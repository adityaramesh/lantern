local optim = lt.optim
local adam, parent = torch.class('lt.adam', 'lt.optimizer_base')

--[[
Required parameters:
* `tensor_type`
* `tensor_size`

Optional Parameters:
* `step_size`: Default is 1e-3.
* `beta_1`:    Default is 0.9.
* `beta_2`:    Default is 0.999.
* `epsilon`:   Default is 1e-8.
* `logger`
--]]
function adam:__init(args)
	parent.__init(self, args)

	self.state.learningRate = args.step_size or 1e-3
	self.state.beta1        = args.beta_1    or 0.9
	self.state.beta2        = args.beta_2    or 0.999
	self.state.epsilon      = args.epsilon   or 1e-8

	self:initialize()
end

function adam:initialize()
	parent.initialize(self)

	if self.state.m ~= nil then
		assert(self.state.v ~= nil)
		assert(self.state.denom ~= nil)
		assert(self.state.m:type() == self.state.v:type())
		assert(self.state.m:type() == self.state.denom:type())

		-- Transfer the parameters back to the GPU after deserialization.
		if self.state.m:type() ~= self.state.tensor_type then
			self.state.m     = self.state.m:type(self.state.tensor_type)
			self.state.v     = self.state.v:type(self.state.tensor_type)
			self.state.denom = self.state.denom:type(self.state.tensor_type)
		end
	end
end

function adam:update(params, eval_func)
	optim.adam(eval_func, params, self.state)
end

function adam:__save(args)
	local temp_m     = self.state.m
	local temp_v     = self.state.v
	local temp_denom = self.state.denom

	if self.state.tensor_type == 'torch.CudaTensor' then
		self.state.m     = self.state.m:float()
		self.state.v     = self.state.v:float()
		self.state.denom = self.state.denom:float()
	end

	parent.__save(self, args)

	if self.state.tensor_type == 'torch.CudaTensor' then
		self.state.m     = temp_m
		self.state.v     = temp_v
		self.state.denom = temp_denom
	end

	collectgarbage()
end
