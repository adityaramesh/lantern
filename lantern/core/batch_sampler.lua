require "cutorch"

local alternating_batch_sampler = lantern.make_class("alternating_batch_sampler")
local sequential_batch_sampler  = lantern.make_class("sequential_batch_sampler")

local function validate_args(args)
	assert(type(args) == "table")
	assert(args.target == "cpu" or args.target == "gpu")
	assert(type(args.data) == "table")
	assert(#args.data >= 1)
	assert(args.batch_size >= 1)
	assert(type(args.shuffle) == "boolean")
	assert(type(args.input_shape) == "userdata")
	assert(type(args.target_shape) == "userdata")
	assert(type(args.input_type) == "string")
	assert(type(args.target_type) == "string")
end

local function initialize(class, args)
	validate_args(args)

	class.data       = args.data
	class.batch_size = args.batch_size

	-- First, get the information necessary to allocate the buffers used to
	-- form the mini-batches.

	local input_batch_type
	local target_batch_type

	if args.target == "cpu" then
		input_batch_type = args.input_type
		target_batch_type = args.target_type
	else
		input_batch_type = "torch.CudaTensor"
		target_batch_type = "torch.CudaTensor"
	end

	local input_batch_shape = torch.LongStorage(#args.input_shape + 1)
	input_batch_shape[1] = args.batch_size

	for i = 1, #args.input_shape do
		input_batch_shape[i + 1] = args.input_shape[i]
	end

	local target_batch_shape = torch.LongStorage(#args.target_shape + 1)
	target_batch_shape[1] = args.batch_size

	for i = 1, #args.target_shape do
		target_batch_shape[i + 1] = args.target_shape[i]
	end

	class.input_buffer = torch.Tensor(input_batch_shape):type(input_batch_type)
	class.target_buffer = torch.Tensor(target_batch_shape):type(target_batch_type)

	-- Define the data access strategy.
	
	class.index = 1
	if args.shuffle then
		class.access_strategy = lantern.shuffled_access_strategy(class.data)
	else
		class.access_strategy = lantern.linear_access_strategy(class.data)
	end
end

function alternating_batch_sampler:__init(args)
	initialize(self, args)
end

function alternating_batch_sampler:next()
	-- Note: under this scheme, all batches with have sizes equal to
	-- precisely `batch_size`; there is no special corner case.
	
	for i = 1, self.batch_size do
		local offset = self.index + i - 1
		local dataset = (offset - 1) % #self.data + 1
		local instance = (offset - dataset) / #self.data + 1
		local new_index = self.access_strategy:index(dataset, instance)

		assert(new_index >= 1)
		assert(new_index <= self.data[dataset].inputs:size(1))

		self.input_buffer[{{i}}]:copy(self.data[dataset].inputs[{{new_index}}])
		self.target_buffer[{{i}}]:copy(self.data[dataset].targets[{{new_index}}])
	end

	self.index = self.index + self.batch_size
	return self.input_buffer, self.target_buffer
end

function sequential_batch_sampler:__init(args)
	initialize(self, args)

	self.cum_sizes = {[0] = 0}
	for i = 1, #self.data do
		self.cum_sizes[i] = self.data[i].inputs:size(1)
	end

	self.cum_sizes = torch.Tensor(self.cum_sizes):cumsum()
	self.dataset = 1
end

function sequential_batch_sampler:next()
	assert(self.dataset <= #self.data)
	
	if self.index + self.batch_size >= self.cum_sizes[self.dataset] then
		-- Note: we assert in the constructor that the size of each
		-- dataset is at least `batch_size + 1`. So we can cross at most
		-- one boundary between datasets as we form a mini-batch.

		local count_1 = self.cum_sizes[self.dataset] - self.index + 1
		assert(count_1 >= 1)
		assert(count_1 <= self.batch_size)

		local count_2 = self.batch_size - count_1
		local base = self.index - self.cum_sizes[self.dataset - 1] - 1

		for i = 1, count_1 do
			self.input_buffer[{{i}}]:copy(self.data[self.dataset].inputs[{{base + i}}])
			self.target_buffer[{{i}}]:copy(self.data[self.dataset].targets[{{base + i}}])
		end

		for i = 1, count_2 do
			self.input_buffer[{{count_1 + i}}]:copy(
				self.data[self.dataset + 1].inputs[{{i}}])
			self.target_buffer[{{count_1 + i}}]:copy(
				self.data[self.dataset + 1].targets[{{i}}])
		end

		self.index = self.index + self.batch_size
		self.dataset = self.dataset + 1
		return self.input_buffer, self.target_buffer
	elseif self.index + self.batch_size > self.cum_sizes[#self.cum_sizes] then
		-- This is the case in which we have to use a slightly smaller
		-- minibatch, so that we do not go past the end of the last
		-- dataset.

		assert(self.dataset == #self.cum_sizes)
		local count = self.cum_sizes[#self.cum_sizes] - self.index + 1
		local base = self.index - self.cum_sizes[#self.cum_sizes - 1] - 1

		for i = 1, count do
			self.input_buffer[{{i}}]:copy(self.data[#self.data].inputs[{{base + i}}])
			self.target_buffer[{{i}}]:copy(self.data[#self.data].targets[{{base + i}}])
		end

		self.index = self.index + self.batch_size
		self.dataset = self.dataset + 1
		return self.input_buffer[{{1, count}}], self.target_buffer[{{1, count}}]
	else
		local base = self.index - self.cum_sizes[#self.dataset - 1] - 1

		for i = 1, self.batch_size do
			self.input_buffer[{{i}}]:copy(self.data[#self.data].inputs[{{base + i}}])
			self.target_buffer[{{i}}]:copy(self.data[#self.data].targets[{{base + i}}])
		end

		self.index = self.index + self.batch_size
		self.dataset = self.dataset + 1
		return self.input_buffer, self.target_buffer
	end
end
