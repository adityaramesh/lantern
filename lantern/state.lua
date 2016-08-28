local cutorch = lt.cutorch
local global_rng_state, parent = torch.class('lt.global_rng_state', 'lt.serializable')

function global_rng_state:__init()
	parent.__init(self)
	torch.manualSeed(0)
	if cutorch then cutorch.manualSeed(0) end
end

function global_rng_state:__save(args)
	self.state.host_rng_state = torch.getRNGState()
	if cutorch then self.state.device_rng_state = cutorch.getRNGState() end
	parent.__save(self, args)
end

function global_rng_state:__load(args)
	parent.__load(self, args)

	if self.state.host_rng_state == nil then
		torch.manualSeed(0)
		if cutorch then cutorch.manualSeed(0) end
	else
		assert(self.state.device_rng_state ~= nil)
		torch.setRNGState(self.state.host_rng_state)
		if cutorch then cutorch.setRNGState(self.state.device_rng_state) end
	end
end
