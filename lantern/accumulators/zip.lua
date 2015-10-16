local zip = lantern.make_accumulator("zip")

function zip:__init(accs)
	assert(type(accs) == "table")
	assert(#accs >= 1)
	self.accs = accs
end

function zip:update(batch, state)
	for _, v in pairs(self.accs) do
		v:update(batch, state)
	end
end

function zip:value()
	local values = {}

	-- Each accumulator owned by this metric returns a table of key-value
	-- pairs. We collect all of these in a single table, and output, the
	-- result.
	for _, v in pairs(self.accs) do
		local output = v:value()
		assert(type(output) == "table")

		for k, v in pairs(output) do
			assert(not values[k])
			values[k] = v
		end
	end
	return values
end
