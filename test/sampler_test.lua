require "torch"
dofile "init.lua"

local function make_dataset()
	local ds_1 = "temp/1.t7"
	local ds_2 = "temp/2.t7"
	local ds_3 = "temp/3.t7"
	local ds_4 = "temp/4.t7"

	local i1 = torch.Tensor{1, 1, 1, 1, 1, 1}
	local t1 = torch.Tensor{1, 1, 1, 1, 1, 1}
	local i2 = torch.Tensor{2, 2, 2, 2}
	local t2 = torch.Tensor{2, 2, 2, 2}
	local i3 = torch.Tensor{3, 3, 3, 3, 3, 3}
	local t3 = torch.Tensor{3, 3, 3, 3, 3, 3}
	local i4 = torch.Tensor{4, 4, 4, 4, 4}
	local t4 = torch.Tensor{4, 4, 4, 4, 4}

	local save = function(ds, i, t)
		torch.save(ds, {inputs = i, targets = t, classes = 4})
	end

	save(ds_1, i1, t1)
	save(ds_2, i2, t2)
	save(ds_3, i3, t3)
	save(ds_4, i4, t4)
end

make_dataset()

local p = lantern.batch_provider({
	train_files       = {"temp/1.t7", "temp/2.t7", "temp/3.t7", "temp/4.t7"},
	target            = "cpu",
	-- Change this to test different sampling strategies.
	sampling_strategy = "alternating",
	shuffle           = false,
	batch_size        = 3
})

-- Should be:
-- * 8 for "mixed" and "alternating"
-- * 7 for "sequential"
print(p.train_batches)

local s = p:make_train_sampler()
for i = 1, p.train_batches do
	local b = s:next()
	print(b.inputs)
	print()
end
