require "nn"

dofile "init.lua"
local classes = 10

local function test_1(acc)
	local layer = nn.LogSoftMax(classes)

	for i = 1, 10 do
		local input = torch.zeros(classes)
		local target = (i - 1) % 10 + 1
		input[target] = 1

		local output = layer:forward(input)
		acc:update(output, target)
	end
	return acc:value()
end

local function test_2(acc)
	local layer = nn.LogSoftMax(classes)

	for i = 1, 10 do
		local input = torch.zeros(2, classes)
		local target = (i - 1) % 10 + 1
		local targets = torch.LongTensor{target, target}
		input[{1, target}] = 1
		input[{2, target}] = 1

		local output = layer:forward(input)
		acc:update(output, targets)
	end
	return acc:value()
end

local acc = lantern.accumulators.accuracy(classes)
print(test_1(acc))
print(test_2(acc))
