--[[
A simple ResNet image classifier.
--]]

--[[
TODO compatibility with non-GPU mode.
--]]

local bit32 = require('bit32')

local nn      = lt.nn
local cunn    = lt.cunn
local cudnn   = lt.cudnn
local cutorch = lt.cutorch

local model, parent = torch.class('model', 'lt.model_base')

--[[
Arguments:
- `input_size`: `LongTensor` describing the size of the input image.
- `class_count`: Number of classes.
--]]
function model:__init(args)
	parent.__init(self, args)

	local function is_integer(x)
		return x == math.floor(x)
	end

	local function is_power_of_two(n)
		assert(is_integer(n))
		return bit32.band(n, n - 1) == 0
	end

	local function round(x)
		return math.floor(x + 0.5)
	end

	local function log_2(n)
		assert(is_integer(n))
		return round(math.log(n) / math.log(2))
	end

	assert(#args.input_size == 3)
	assert(args.class_count >= 2)

	local channel_count = args.input_size[1]
	local image_width   = args.input_size[2]
	local class_count   = args.class_count

	assert(channel_count >= 1)
	assert(image_width == args.input_size[3])
	assert(image_width >= 8 and is_power_of_two(image_width))

	local try_cudnn = function(class)
		if cudnn and cudnn[class] then return cudnn[class] end
		return nn[class]
	end

	local conv = function(...)
		local m = try_cudnn('SpatialConvolution')(...)
		m.bias, m.gradBias = nil, nil
		return m
	end

	local pool        = try_cudnn('SpatialMaxPooling')
	local bn          = try_cudnn('SpatialBatchNormalization')
	local log_softmax = try_cudnn('LogSoftMax')

	local function make_block(n_in, n_out, is_input_block)
		if is_input_block == nil then
			is_input_block = false
		end

		local dw_in
		local main_path

		if n_in == n_out then
			dw_in = 1
			main_path = nn.Identity()
		else
			dw_in = 2
			main_path = nn.Sequential()
				:add(pool(2, 2, 2, 2))
				:add(conv(n_in, n_out, 1, 1))
		end

		local residual_path = nn.Sequential()

		if not is_input_block then
			residual_path
				:add(bn(n_in))
				:add(nn.LeakyReLU(0.2, true))
		end

		residual_path
			:add(conv(n_in, n_out, 3, 3, 1, 1, 1, 1))
			:add(bn(n_out))
			:add(nn.LeakyReLU(0.2, true))
			:add(conv(n_out, n_out, 3, 3, 2, 2, 1, 1))

		return nn.Sequential()
			:add(nn.ConcatTable()
				:add(main_path)
				:add(residual_path))
			:add(nn.CAddTable(true))
	end

	local make_output_block = function(n_in)
		return nn.Sequential()
			:add(bn(n_in))
			:add(nn.LeakyReLU(0.2, true))
			:add(conv(n_in, class_count, 4, 4))
			:add(nn.View(-1, class_count))
			--:add(log_softmax())
	end

	local fm_count    = 64
	local block_count = log_2(image_width) - 1
	assert(block_count >= 2)

	local m = nn.Sequential()
	m:add(make_block(channel_count, fm_count, true))

	for i = 1, block_count - 2 do
		m:add(make_block(2^(i - 1) * fm_count, 2^i * fm_count))
	end

	m:add(make_output_block(2^(block_count - 2) * fm_count))

	self.state.module = m
	parent.initialize(self)
	if cutorch then m:cuda() end
end

function model:notify_post_update()
	--[[
	If the model needs to do things like zero out or renormalize certain parameters after being
	updated, then this behavior should be implemented here.
	--]]
end
