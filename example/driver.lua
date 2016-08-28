--[[
TODO: check that this works in CPU and GPU mode.
--]]

local F       = lt.F
local cutorch = lt.cutorch

local nn    = require('nn')
local cunn  = require('cunn')
local cudnn = require('cudnn')

if cudnn then
	cudnn.benchmark = true
	cudnn.fastest = true
	--cudnn.verbose = true
end

local function make_evaluator(args)
	return function(_)
		args.model:grad_parameters():zero()

		local preds     = args.model:forward(args.images)
		local loss      = args.crit:forward(args.images, args.labels)
		local grad_loss = args.crit:backward(args.images, args.labels)

		args.model:backward(args.images, grad_loss)
		return nil, args.model:grad_parameters()
	end
end

--[[
TODO: document the arguments that this function accepts.
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

	local data_size        = args.data.inputs:size()
	local batch_size       = args.batch_size       or 64
	local max_epochs       = args.max_epochs       or 10000
	local iters_per_epoch  = args.iters_per_epoch  or math.ceil(data_size[1] / batch_size)
	local iters_per_sync   = args.iters_per_sync   or math.floor(iters_per_epoch / 100)

	assert(iters_per_sync >= 2 and iters_per_sync <= iters_per_epoch)

	local cp = lt.checkpointer{
		output_dir = checkpoint_dir,
		objects    = {'model', 'optimizer', 'global_rng_state', 'logger'},
		logger     = tl,
	}

	local image_size = torch.LongStorage(#data_size - 1)
	for i = 2, #data_size do image_size[i - 1] = data_size[i] end

	-- Note: the state should be constructed before everything else, so that any use of random
	-- number generation (e.g. to initialize weights) is reproducible.
	local state  = cp.global_rng_state or lt.global_rng_state{}
	local model  = cp.model or args.make_model{input_size = image_size, logger = tl}
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
	                           	step_size   = 2e-4,
					beta_1      = 0.5,
	                           	tensor_type = params:type(),
	                            	tensor_size = params:size(),
	                            	logger      = tl
	                    }

	local crit = nn.ClassNLLCriterion()
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
		}
	}

	local cur_epoch 
	if cp:last_epoch() then cur_epoch = cp:last_epoch() + 1
	else cur_epoch = 1 end

	local tensor_type
	if cutorch then tensor_type = torch.CudaTensor
	else tensor_type = torch.FloatTensor end

	local make_buffer = function(data)
		local size = data:size()
		size[1] = batch_size
		return tensor_type(size)
	end

	local image_buffer = make_buffer(args.data.inputs)
	local label_buffer = make_buffer(args.data.targets)

	local eval_args = {
		model      = model,
		crit       = crit,
		opt        = opt,
		logger     = tl,
		batch_size = batch_size,

		images = image_buffer,
		labels = label_buffer,
		iter_counter = 1,
	}

	local eval = make_evaluator(eval_args)
	model:training()

	for epoch = cur_epoch, max_epochs do
		tl:log('/console/info', F"Starting epoch {epoch}.")

		-- TODO make data sampler

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

			-- TODO invoke data sampler (make this actually do something meaningful)
			--sampler()

			if sync then
				cutorch.synchronize()
				jl.data_time = iter_timer:time().real
				iter_timer:reset()
				iter_timer:resume()
			end

			opt:update(model_params, eval)

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
		jl.epoch_time = epoch_timer:time().real
		jl:commit()
		jl:flush()

		cp:update(epoch, {
			model  = model,
			opt    = opt,
			state  = state,
			logger = jl,
		})
	end
end
