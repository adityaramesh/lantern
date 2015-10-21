require "torch"

local gradient_norm = lantern.make_accumulator("gradient_norm")

function gradient_norm:__init(model)
	self.name  = "gradient_norm"
	self.model = model
	self.entries = {}
end

function gradient_norm:update(batch, state)
	self.entries[#self.entries + 1] = self.model:grad_parameters():norm()
end

function gradient_norm:value()
	assert(#self.entries > 0)

	local temp = torch.Tensor(self.entries)

	local std
	if temp:size(1) > 1 then
		std = temp:std()
	else
		-- If we call `:std()` in this case, we will get NaN.
		std = 0
	end

	return {
		mean = temp:mean(),
		std  = std,
		max  = temp:max(),
		min  = temp:min()
	}
end
