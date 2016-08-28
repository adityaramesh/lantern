require('lt')
require('example/driver')

local F = lt.F
local hdf5 = require('hdf5')

local function extra_options(cmd)
	cmd:option('-train_data_file', 'data/mnist/padded/train.hdf5')
	cmd:option('-test_data_file',  'data/mnist/padded/test.hdf5')
	cmd:option('-model_source',    'example/model.lua')
	cmd:option('-max_epochs',      300)
	cmd:option('-class_count',     10)
end

local options = lt.parse_options(extra_options)

for _, file in pairs{options.train_data_file, options.test_data_file, options.model_source} do
	lt.fail_if(not paths.filep(file), F"File '{file}' does not exist.")
end

dofile(options.model_source)
local train_data   = hdf5.open(options.train_data_file):read():all()
local test_data    = hdf5.open(options.test_data_file):read():all()
local train_inputs = train_data.inputs

local function reshape_if_required(inputs)
	if inputs:nDimension() == 3 then
		inputs:resize(inputs:size(1), 1, inputs:size(2), inputs:size(3))
	end
end

reshape_if_required(train_data.inputs)
reshape_if_required(test_data.inputs)

lt.run{
	gpu            = options.gpu,
	max_epochs     = options.max_epochs,
	experiment_dir = options.experiment_dir,
	train_data     = train_data,
	test_data      = test_data,
	class_count    = options.class_count,

	make_model = function(args) return model(args) end,
}
