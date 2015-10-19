--
-- The `batch_provider` class is used to generate objects that can be used
-- sample from one or more data sources during training or testing.
--

local batch_provider = lantern.make_class("batch_provider")

local function validate_args(args)
	assert(type(args) == "table")
	assert(args.target == "cpu" or args.target == "gpu")

	if args.train_files then
		assert(type(args.train_files) == "table")
		assert(#args.train_files >= 1)
	end

	if args.test_file then
		assert(type(args.test_file) == "string")
	end

	if args.batch_size then
		assert(args.batch_size >= 1)
	else
		args.batch_size = 1
	end

	if args.shuffle then
		assert(type(args.shuffle) == "boolean")
	else
		args.shuffle = true
	end

	if args.sampling_strategy then
		assert(
			args.sampling_strategy == "mixed"       or
			args.sampling_strategy == "alternating" or
			args.sampling_strategy == "sequential"
		)
	else
		args.sampling_strategy = "mixed"
	end

	if not args.logger then
		args.logger = lantern.stdout_logger()
	end
end

--
-- The arguments to this function must be provided in a table.
--
-- Parameters:
-- * train_files (required for training): A table containing one or more strings
--   representing the paths to the files to be used for training data.
-- * test_file (required for testing): The path to the file to be used for
--   testing.
-- * target (required): Either "cpu" or "gpu".
-- * batch_size (optional): The size of the batches returned by the samplers.
--   Default: 1.
-- * sampling_strategy (optional): The strategy used to determine how to make
--   use of multiple training files. Supported options are as follows. The
--   default is "mixed".
--   * mixed: Each batch consists of a mixture of instances sampled from the
--     training datasets in a round-robin fashion. When we reach the end of a
--     dataset duirng an epoch, we wrap back to the beginning.
--   * alternating: Each batch consists only of instances from a single
--     dataset, but we alternate between datasets while sampling. After we reach
--     the end of a dataset during an epoch, we wrap back to the beginning.
--   * sequential: Samples all instances from one file before moving on to the
--     next. A batch may contain instances from two datasets.
-- * shuffle (optional): Indicates whether the data in the training files should
--   be accessed in a random order that is determined at the start of each
--   training epoch. Default: `true`.
-- * logger (optional): Used to log real-time events. Default:
--   `lantern.stdout_logger`.
--
function batch_provider:__init(args)
	validate_args(args)

	self.train_files       = args.train_files
	self.test_file         = args.test_file
	self.batch_size        = args.batch_size
	self.target            = args.target
	self.sampling_strategy = args.sampling_strategy
	self.shuffle           = args.shuffle
	self.logger            = args.logger

	if self.train_files then
		self:load_train_data()
	end

	if self.test_file then
		self:load_test_data()
	end

	if self.train_data then
		self:infer_properties(self.train_data[1])
	else
		self:infer_properties(self.test_data)
	end
end

function batch_provider:load_train_data()
	self.train_data = {}
	self.max_train_size = 0
	self.total_train_size = 0

	for _, v in pairs(self.train_files) do
		local data = lantern.load(v)
		local size = data.inputs:size(1)
		assert(size == data.targets:size(1))

		-- If the batch size is greater than or equal to the size of one
		-- of the data sets, then there will be a huge amount of
		-- redundancy in the generated batches. The training procedure
		-- is at the very least highly questionable. I think it makes
		-- sense to fail in this case.
		assert(
			size > self.batch_size,
			"Size of dataset is less than or equal to batch size."
		)

		self.train_data[#self.train_data + 1] = data
		self.max_train_size = math.max(self.max_train_size, size)
		self.total_train_size = self.total_train_size + size
	end

	local pred = function(a, b)
		return a.inputs:size(1) > b.inputs:size(1)
	end

	table.sort(self.train_data, pred)

	self.train_sizes = {}
	for _, data in pairs(self.train_data) do
		self.train_sizes[#self.train_sizes + 1] = data.inputs:size(1)
	end

	if self.sampling_strategy == "mixed" then
		-- Here's the picture used to derive this:
		--
		-- Dataset 1   Dataset 2   Dataset 3
		-- 1 --------> 1 --------> 1
		-- 2 --------> 2 --------> 2
		-- 3 --------> 3 --------> 3
		-- 4 --------> 1 --------> 4
		-- 5 --------> 2 --------> 1
		--
		-- Each time we move "past the end" of a dataset, we wrap back
		-- to the beginning. To compute the minimum number of batches
		-- required to encounter each instance at least once, we need
		-- to compute the number of datasets that are of maximal size.

		local max_size = self.train_sizes[1]
		local max_entries = 1

		for i = 2, #self.train_sizes do
			if self.train_sizes[i] == max_size then
				max_entries = max_entries + 1
			else
				assert(self.train_sizes[i] < max_size)
				break
			end
		end
		
		self.instances = (max_size - 1) * #self.train_data + max_entries
		self.train_batches = math.ceil(self.instances / self.batch_size)
	elseif self.sampling_strategy == "alternating" then
		local max_batches = math.ceil(self.train_sizes[1] / self.batch_size)
		local max_entries = 1

		for i = 2, #self.train_sizes do
			local batches = math.ceil(self.train_sizes[i] / self.batch_size)
			if batches == max_batches then
				max_entries = max_entries + 1
			else
				assert(batches < max_batches)
				break
			end
		end
		
		self.train_batches = (max_batches - 1) * #self.train_data + max_entries
	elseif self.sampling_strategy == "sequential" then
		self.train_batches = math.ceil(self.total_train_size / self.batch_size)
	else
		error("Invalid sampling strategy `" .. self.sampling_strategy .. "`.")
	end
end

function batch_provider:load_test_data()
	self.test_data    = lantern.load(self.test_file)
	self.test_size    = self.test_data.inputs:size(1)
	self.test_batches = math.ceil(self.test_size / self.batch_size)
end

function batch_provider:infer_properties(data)
	self.input_type = data.inputs:type()
	self.target_type = data.targets:type()

	if self.input_type == "torch.DoubleTensor" and self.target == "gpu" then
		self.logger:update(
			"/console/warning", 
			"Input type is double-precision, but will be truncated " ..
			"to single-precision after being transferred to the GPU."
		)
	end

	if self.target_type == "torch.DoubleTensor" and self.target == "gpu" then
		self.logger:update(
			"/console/warning", 
			"Input type is double-precision, but will be truncated " ..
			"to single-precision after being transferred to the GPU."
		)
	end

	if data.inputs:nDimension() == 1 then
		self.input_shape = torch.LongStorage()
	else
		self.input_shape = data.inputs[1]:size()
	end

	if data.targets:nDimension() == 1 then
		self.target_shape = torch.LongStorage()
	else
		self.target_shape = data.targets[1]:size()
	end
end

function batch_provider:make_sampler(mode)
	local data, shuffle, strategy, batches

	if mode == "train" then
		data     = self.train_data
		shuffle  = self.shuffle
		strategy = self.sampling_strategy
		batches  = self.train_batches
	else
		-- In testing mode, there's no reason to use shuffling or the
		-- mixed or alternating sampling strategy.

		data     = {self.test_data}
		shuffle  = false
		strategy = "sequential"
		batches  = self.test_batches
	end
	
	local args = {
		data         = data,
		batch_size   = self.batch_size,
		target       = self.target,
		shuffle      = shuffle,
		input_shape  = self.input_shape,
		target_shape = self.target_shape,
		input_type   = self.input_type,
		target_type  = self.target_type
	}

	if strategy == "mixed" then
		args.instances = self.instances
		return lantern.mixed_batch_sampler(args)
	elseif strategy == "alternating" then
		args.batches = batches
		return lantern.alternating_batch_sampler(args)
	elseif strategy == "sequential" then
		return lantern.sequential_batch_sampler(args)
	else
		error("Invalid sampling strategy `" .. self.sampling_strategy .. "`.")
	end
end

function batch_provider:make_train_sampler()
	return self:make_sampler("train")
end

function batch_provider:make_test_sampler()
	return self:make_sampler("test")
end
