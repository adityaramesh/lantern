--
-- The `driver` class contains the default implementations of the training and
-- testing epoch functions.
--

require "sys"
local driver = lantern.make_class("driver")

--
-- The only argument to the constructor is the `batch_provider` instance used to
-- construct mini-batches from the data.
--
function driver:__init(bp)
	self.bp = bp

	if bp.train_data then
		self.train = true
	else
		self.train = false
	end

	if bp.test_data then
		self.test = true
	else
		self.test = false
	end
end

function driver:train_epoch(model, optim, acc, logger)
	assert(self.train)
	local sampler = self.bp:make_train_sampler()

	for i = 1, self.bp.train_batches do
		local start = sys.clock()
		local inputs, targets = sampler:next()
		local outputs, loss = optim:update(inputs, targets)
		acc:update(outputs, loss, targets)

		logger:update("/progress", {
			processed_instances = i,
			total_instances = self.bp.train_batches,
			time = sys.clock() - start
		})
	end
	return acc:value()
end

function driver:test_epoch(model, optim, acc, logger)
	assert(self.test)
	local sampler = self.bp:make_test_sampler()

	for i = 1, self.bp.test_batches do
		local start = sys.clock()
		local inputs, targets = sampler:next()
		local outputs, loss = model:forward(inputs, targets)
		acc:update(outputs, loss, targets)

		logger:update("/progress", {
			processed_instances = i,
			total_instances = self.bp.test_batches,
			time = sys.clock() - start
		})
	end
	return acc:value()
end
