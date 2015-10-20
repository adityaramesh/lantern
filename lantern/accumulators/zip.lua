local zip = lantern.make_accumulator("zip")

function zip:__init(accs)
	assert(type(accs) == "table")
	assert(#accs >= 1)

	self.name = "zip"
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
	for _, acc in pairs(self.accs) do
		local output = acc:value()
		assert(type(output) == "table")

		local count = 0
		for _, _ in pairs(output) do
			count = count + 1
		end

		assert(count > 0)
		assert(not values[acc.name])

		if count == 1 and output[acc.name] then
			values[acc.name] = output[acc.name]
		else
			values[acc.name] = output
		end
	end
	return values
end
