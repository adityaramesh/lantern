--[[
TODO: check that this works in CPU and GPU mode.
--]]

local F       = lt.F
local nn      = lt.nn
local cunn    = lt.cunn
local cudnn   = lt.cudnn
local cutorch = lt.cutorch

require('example/data_sampler')

local function make_evaluator(args)
	return function(_)
		args.model:grad_parameters():zero()
		local input_count = args.images:size(1)

		local preds     = args.model:forward(args.images)
		local loss      = args.crit:forward(preds, args.labels)
		local grad_loss = args.crit:backward(preds, args.labels)

		_, pred_classes = torch.max(preds, 2)

		args.total_correct = args.total_correct + torch.sum(torch.eq(
			pred_classes:view(args.batch_size), args.labels))
		args.total_seen = args.total_seen + input_count

		args.model:backward(args.images, grad_loss)
		args.model:grad_parameters():div(input_count)
		return loss, args.model:grad_parameters()
	end
end

local function compute_test_accuracy(args)
	local input_count = args.images:size(1)
	local batch_size  = args.batch_size
	local iter_count  = math.ceil(input_count / batch_size)

	local total_seen = 0
	local total_correct = 0

	for iter = 1, iter_count do
		local a = (iter - 1) * batch_size + 1
		local b = math.min(input_count, iter * batch_size)
		local count = b - a + 1

		local inputs = args.image_buffer[{{1, count}}]
		local targets = args.label_buffer[{{1, count}}]

		inputs:copy(args.images[{{a, b}}])
		targets:copy(args.labels[{{a, b}}])

		local preds = args.model:forward(inputs)
		_, pred_classes = torch.max(preds, 2)

		total_correct = total_correct + torch.sum(torch.eq(
			pred_classes:view(count), targets))
		total_seen = total_seen + count

		args.logger:log('/progress', {current = iter, total = iter_count})
	end

	assert(total_seen == input_count)
	return total_correct / total_seen
end

--[[
Required arguments:
- `experiment_dir`: Path to the directory in which to save the current results.
- `train_data`: Table with two keys, `inputs` and `targets`.
- `test_data`: Table with two keys, `inputs` and `targets`.
- `class_count`: Number of classes.
- `make_model`: Function that returns a newly-created model when given the input size and logger.
- `gpu`: GPU ordinal (must be greater than or equal to one).

Optional arguments:
- `batch_size`: Defaults to 64.
- `max_epochs`: Defaults to 10,000.
- `iters_per_epoch`: Defaults to `math.ceil(data_size / batch_size)`.
- `iters_per_sync`: Defaults to `math.floor(iters_per_epoch / 100)`.
- `iters_per_val`: Defaults to five.
- `make_optimizer`: Defaults to Adam.
--]]
function lt.run(args)
	local tl = args.logger or lt.tty_logger()
	local experiment_dir = args.experiment_dir

	assert(type(experiment_dir) == 'string')
	assert(paths.dirp(experiment_dir))

	local output_dir     = paths.concat(experiment_dir, 'output')
	local checkpoint_dir = paths.concat(experiment_dir, 'checkpoint')

	for _, dir in pairs{output_dir, checkpoint_dir} do
		lt.make_directory_if_not_exists(dir, tl)
	end

	if cutorch then cutorch.setDevice(args.gpu) end

	local data_size        = args.train_data.inputs:size()
	local batch_size       = args.batch_size      or 64
	local max_epochs       = args.max_epochs      or 10000
	local iters_per_epoch  = args.iters_per_epoch or math.ceil(data_size[1] / batch_size)
	local iters_per_sync   = args.iters_per_sync  or math.floor(iters_per_epoch / 100)
	local iters_per_val    = args.iters_per_val   or 5

	assert(iters_per_sync >= 2 and iters_per_sync <= iters_per_epoch)

	local cp = lt.checkpointer{
		output_dir = checkpoint_dir,
		objects    = {'model', 'optimizer', 'global_rng_state', 'logger'},
		logger     = tl,
	}

	local image_size = torch.LongStorage(#data_size - 1)
	for i = 2, #data_size do image_size[i - 1] = data_size[i] end

	assert(args.class_count >= 2)

	-- Note: the state should be constructed before everything else, so that any use of random
	-- number generation (e.g. to initialize weights) is reproducible.
	local state  = cp.global_rng_state or lt.global_rng_state{}
	local model  = cp.model or args.make_model{input_size = image_size,
		class_count = args.class_count, logger = tl}
	local params = model:parameters()

	-- Invokes a factory function if it was provided by the user, and supplies extra arguments
	-- from this driver.
	local invoke_if_given = function(f, extra_args)
		if f then return f(extra_args) end
	end

	local opt = cp.optimizer or invoke_if_given(args.make_optimizer, {
	                          	tensor_type = params:type(),
	                          	tensor_size = params:size(),
					logger      = tl
	                    }) or lt.adam{
	                           	step_size   = 1e-3,
	                           	tensor_type = params:type(),
	                            	tensor_size = params:size(),
	                            	logger      = tl
	                    }

	local crit = nn.CrossEntropyCriterion()
	if cutorch then crit:cuda() end

	local jl = cp.logger or lt.json_logger{
		output_dir = output_dir,
		fields = {
			'epoch',
			'iteration',
			'loss',

			'data_time',
			'update_time',
			'avg_iter_time',
			'epoch_time',

			'train_acc',
			'test_acc',
		}
	}

	local cur_epoch 
	if cp:last_epoch() then cur_epoch = cp:last_epoch() + 1
	else cur_epoch = 1 end

	local make_buffer_like = function(buf)
		local tensor_type
		local t = buf:type()

		if t == 'torch.FloatTensor' then
			if cutorch then tensor_type = torch.CudaTensor
			else tensor_type = torch.DoubleTensor end
		else
			if cutorch then tensor_type = torch.CudaLongTensor
			else tensor_type = torch.LongTensor end
		end

		local size = buf:size()
		size[1] = batch_size
		return tensor_type():resize(size)
	end

	local image_buffer = make_buffer_like(args.train_data.inputs)
	local label_buffer = make_buffer_like(args.train_data.targets)

	local train_args = {
		model      = model,
		crit       = crit,
		opt        = opt,
		logger     = tl,
		batch_size = batch_size,

		images = image_buffer,
		labels = label_buffer,

		total_correct = 0,
		total_seen = 0,
	}

	local test_args = {
		model      = model,
		logger     = tl,
		batch_size = batch_size,

		images = args.test_data.inputs,
		labels = args.test_data.targets,

		image_buffer = image_buffer,
		label_buffer = label_buffer,
	}

	local eval = make_evaluator(train_args)
	model:training()

	for epoch = cur_epoch, max_epochs do
		tl:log('/console/info', F"Starting epoch {epoch}.")

		local sampler = data_sampler(args.train_data.inputs, args.train_data.targets,
			image_buffer, label_buffer, iters_per_epoch)

		local epoch_timer = torch.Timer()
		local iter_timer  = torch.Timer()
		local timed_iters = 0

		iter_timer:stop()
		iter_timer:reset()

		for iter = 1, iters_per_epoch do
			jl.epoch = epoch
			jl.iteration = iter
			local sync = iter >= 2 and (iter - 1) % iters_per_sync == 0

			if sync then
				assert(timed_iters >= 2)
				cutorch.synchronize()
				iter_timer:stop()

				jl.avg_iter_time = iter_timer:time().real / timed_iters
				timed_iters = 0
			end

			if sync then
				iter_timer:reset()
				iter_timer:resume()
			end

			sampler()

			if sync then
				cutorch.synchronize()
				jl.data_time = iter_timer:time().real
				iter_timer:reset()
				iter_timer:resume()
			end

			opt:update(params, eval)

			--[[
			In case the model needs to do things like zero out or renormalize certain
			parameters after being updated.
			--]]
			model:notify_post_update()

			if sync then
				cutorch.synchronize()
				iter_timer:stop()
				jl.update_time = iter_timer:time().real

				--[[
				If the GPU computation overlaps with IO, then the the average time
				per iteration should not be much larger than the maximum of the data
				time and update time.
				--]]
				local slack    = 1.2
				local avg_time = jl.avg_iter_time
				local thres    = slack * math.max(jl.data_time, jl.update_time)

				-- Uncomment for easy performance debugging.
				-- print(jl.avg_iter_time, jl.data_time, jl.update_time)

				if avg_time >= thres then
					tl:log('/console/warning', F[[
						Average time per iteration ({1000 * avg_time} ms) 
						is greater than {slack} times the maximum of the 
						data sampling time and the model update time ({1000
						* thres} ms). This may mean that there is unexpected
						 GPU synchronization in the training procedure.
					]])
				end

				if iter ~= iters_per_epoch then
					jl:commit()
					jl:flush()
				end

				iter_timer:reset()
				iter_timer:resume()
			else
				if iter ~= iters_per_epoch then
					jl:commit()
				end

				if iter == 2 then
					iter_timer:reset()
					iter_timer:resume()
				else
					timed_iters = timed_iters + 1
				end
			end

			tl:log('/progress', {current = iter, total = iters_per_epoch})
		end

		epoch_timer:stop()

		local train_acc = train_args.total_correct / train_args.total_seen
		train_args.total_correct = 0
		train_args.total_seen = 0

		jl.epoch_time = epoch_timer:time().real
		jl.train_acc = train_acc
		tl:log('/console/info', F"Train accuracy: {100 * train_acc}%.")

		if epoch == 1 or (epoch - 1) % iters_per_val == 0 then
			tl:log('/console/info', "Starting test epoch.")

			model:evaluate()
			local test_acc = compute_test_accuracy(test_args)
			model:training()

			jl.test_acc = test_acc
			tl:log('/console/info', F"Test accuracy: {100 * test_acc}%.")
		end

		jl:commit()
		jl:flush()

		cp:update(epoch, {
			model            = model,
			optimizer        = opt,
			global_rng_state = state,
			logger           = jl,
		})
	end
end
