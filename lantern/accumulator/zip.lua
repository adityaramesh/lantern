local zip = torch.class('lantern.accumulator.zip')

function zip:__init(accs)
	assert(type(accs) == 'table')
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

	-- Each accumulator owned by this one returns a table of key-value pairs. We collect the
	-- metrics of the children accumulators in a single table, and output the result.
	for _, acc in pairs(self.accs) do
		local output = acc:value()
		assert(type(output) == 'table')

		local count = 0
		for _, _ in pairs(output) do
			count = count + 1
		end

		assert(count > 0)
		assert(not values[acc.name])

		-- If the output of an accumulator only contains a single key-value pair with the
		-- key being the name of the accumulator, then we only collect the value.
		if count == 1 and output[acc.name] then
			values[acc.name] = output[acc.name]
		else
			values[acc.name] = output
		end
	end

	return values
end
