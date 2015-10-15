--
-- Test cases:
-- * Number of training files (1, 2, 3), all different sizes.
-- * Batch size (sizes that do and do not divide training sizes).
-- * Alternate or sequential sampling.
-- * Shuffling or no shuffling.
--

dofile "init.lua"

local args = {
	train_files = {"data/mnist/raw/train_32x32.t7"},
	test_file   = "data/mnist/raw/test_32x32.t7",
	target      = "gpu",
	batch_size  = 200
}

local p = lantern.batch_provider(args)
local s = p:make_train_sampler()
for i = 1, 10 do
	local a, b = s:next()
	print(a:size())
	print(b:size())
end
