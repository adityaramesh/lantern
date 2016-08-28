local data_sampler = torch.class('data_sampler')

function data_sampler:__init(inputs, targets, input_buffer, target_buffer, iter_count)
	self.inputs        = inputs
	self.targets       = targets
	self.input_buffer  = input_buffer
	self.target_buffer = target_buffer

	self.data_size = inputs:size(1)
	assert(self.data_size == targets:size(1))

	self.batch_size = input_buffer:size(1)
	assert(self.batch_size == target_buffer:size(1))
	assert(type(self.batch_size) == 'number' and self.batch_size >= 1 and
		self.batch_size <= self.data_size)

	self.cur_iter = 1
	self.iter_count = self.iter_count or self.data_size
	assert(type(self.iter_count) == 'number' and self.iter_count >= 1)

	self.indices = torch.randperm(self.data_size)
end

function data_sampler:__call()
	local a = self.batch_size * (self.cur_iter - 1) + 1
	local b = self.batch_size * self.cur_iter

	a = (a - 1) % self.data_size + 1
	b = (b - 1) % self.data_size + 1
	assert(a ~= b)

	if a < b then
		for j = 1, self.batch_size do
			self.input_buffer[j]:copy(self.inputs[self.indices[a + j - 1]])
			self.target_buffer[j] = self.targets[self.indices[a + j - 1]]
		end
	else
		local count = self.data_size - a + 1

		for j = 1, count do
			self.input_buffer[j]:copy(self.inputs[self.indices[a + j - 1]])
			self.target_buffer[j] = self.targets[self.indices[a + j - 1]]
		end

		for j = 1, self.batch_size - count do
			self.input_buffer[count + j]:copy(self.inputs[self.indices[j]])
			self.target_buffer[count + j] = self.targets[self.indices[j]]
		end
	end

	self.cur_iter = self.cur_iter + 1
end
