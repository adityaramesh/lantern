require "torch"

local gradient_norm = lantern.make_accumulator("gradient_norm")

function gradient_norm:__init(model)
	self.model = model
	self.entries = {}
end

function gradient_norm:update(batch, state)
	self.entries[#self.entries + 1] = model:grad_parameters():norm()
end

function gradient_norm:value()
	assert(#self.entries > 0)

	local temp = torch.Tensor(self.entries)
	return {
		mean = temp:mean(),
		std  = temp:std()
	}
end
