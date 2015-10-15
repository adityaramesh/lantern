--
-- Test cases:
-- * Number of training files (1, 2, 3), all different sizes.
-- * Batch size (sizes that do and do not divide training sizes).
-- * Alternate or sequential sampling.
-- * Shuffling or no shuffling.
--

require "sys"
dofile "init.lua"

local train_files = {
	"data/mnist/partitioned/train_left.t7",
	"data/mnist/partitioned/train_right.t7"
}

local targets     = {"cpu", "gpu"}
local test_file   = "data/mnist/scaled/test_32x32.t7"
local strategies  = {"alternating", "sequential"}
local batch_sizes = {1, 2, 3, 59, 200}

-- There are a lot of assertions within the batch sampling classes. For
-- simplicity, we assume that if the code works without any failed assertions,
-- then it is correct.
local function test_config(args)
	local p = lantern.batch_provider(args)
	local s = p:make_train_sampler()
	for i = 1, p.train_batches do
		s:next()
	end
end

local function run_tests(train_files_)
	for _, t in pairs(targets) do
		for _, s in pairs(strategies) do
			for _, bs in pairs(batch_sizes) do
				local args = {
					train_files = train_files_,
					test_file   = test_file,
					target      = t,
					strategy    = s,
					batch_size  = bs
				}

				print("Testing configuration:")
				print(args)
				test_config(args)
			end
		end
	end
end

run_tests({train_files[1]})
run_tests(train_files)
