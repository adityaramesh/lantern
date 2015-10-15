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
			args.sampling_strategy == "alternating" or
			args.sampling_strategy == "sequential"
		)
	else
		args.sampling_strategy = "alternating"
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
--   use of multiple training files. Supported options: "alternating" (sample
--   one instance from each file in a round-robin fashion), and "sequential"
--   (sample all instances from one file before moving on to the next). Default:
--   "alternating".
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
	self.train_sizes = {}
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
		assert(size > self.batch_size)

		self.train_sizes[#self.train_sizes + 1] = size
		self.max_train_size = math.max(self.max_train_size, size)
		self.total_train_size = self.total_train_size + size

		self.train_data[#self.train_data + 1] = data
	end

	if self.sampling_strategy == "alternating" then
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
		-- to the beginning. Thus the total number of instances we pass
		-- before getting to to the last instance of the largest dataset
		-- is the numerator of the expression below.
		self.train_batches = math.ceil(
			((self.max_train_size - 1) * #self.train_data + 1) /
			self.batch_size
		)
	else
		self.train_batches = math.floor(self.total_train_size / self.batch_size)
		if self.total_train_size % self.batch_size ~= 0 then
			self.train_batches = self.train_batches + 1
		end
	end
end

function batch_provider:load_test_data()
	self.test_data = lantern.load(self.test_file)
	self.test_size = self.test_data.inputs:size(1)

	self.test_batches = self.test_size / self.batch_size
	if self.test_size % self.batch_size ~= 0 then
		self.test_batches = self.test_batches + 1
	end
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

function batch_provider:make_sampler(data)
	local args = {
		data         = data,
		batch_size   = self.batch_size,
		target       = self.target,
		shuffle      = self.shuffle,
		input_shape  = self.input_shape,
		target_shape = self.target_shape,
		input_type   = self.input_type,
		target_type  = self.target_type
	}

	if self.sampling_strategy == "alternating" then
		return lantern.alternating_batch_sampler(args)
	else
		return lantern.sequential_batch_sampler(args)
	end
end

function batch_provider:make_train_sampler()
	return self:make_sampler(self.train_data)
end

function batch_provider:make_test_sampler()
	return self:make_sampler({self.test_data})
end
