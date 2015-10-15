local zip = lantern.make_accumulator("zip")

function zip:__init(accs)
	assert(type(accs) == "table")
	assert(#accs >= 1)
	self.accs = accs
end

function zip:update(outputs, targets)
	for _, v in self.accs do
		v:update(outputs, targets)
	end
end

function zip:value()
	local values = {}
	for _, v in self.accs do
		values[#values + 1] = v:value()
	end
	return values
end
