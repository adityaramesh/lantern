require "torch"
require "cutorch"

local accuracy = lantern.make_accumulator("accuracy")

function accuracy:__init(classes)
	assert(
		classes > 1,
		"Number of classes must be greater than one. For binary " ..
		"classification, use the indices one and two."
	)

	self.classes = classes
	self.correct = 0
	self.total   = 0
end

-- Note: each row of outputs should contain the probabilities or log
-- probabilities of the classes.
function accuracy:update(batch, state)
	local targets = batch.targets
	local outputs = state.outputs
	assert(targets)
	assert(outputs)

	if type(targets) ~= "number" then
		local t = targets:type()
		assert(
			t == "torch.ByteTensor" or
			t == "torch.LongTensor" or
			t == "torch.CudaTensor"
		)
	end

	assert(
		outputs:nDimension() <= 2,
		"`outputs` must either be a vector or a matrix whose rows " ..
		"contain the log probabilities of the classes for each input."
	)

	local check_target = function(target)
		assert(
			target > 0,
			"Found target of zero. The target must be the " ..
			"_one-based_ index of the ground-truth class."
		)
		assert(
			target <= self.classes,
			"Found target greater than number of classes given " ..
			"to constructor."
		)
	end

	if outputs:nDimension() == 1 then
		assert(type(targets) == "number")
		check_target(targets)

		local _, indices = torch.max(outputs, 1)
		self.total = self.total + 1

		if indices[1] == targets then
			self.correct = self.correct + 1
		end
	else
		assert(targets:nDimension() == 1)
		assert(outputs:size(1) == targets:size(1))
		for i = 1, targets:size(1) do
			check_target(targets[i])
		end

		local _, indices = torch.max(outputs, 2)
		self.total = self.total + outputs:size(1)
		self.correct = self.correct + torch.eq(indices, targets:typeAs(indices)):sum()
	end

	assert(self.total > 0)
	assert(self.total >= self.correct)
end

function accuracy:value()
	assert(self.total > 0)
	assert(self.total >= self.correct)
	return {accuracy = self.correct / self.total}
end
