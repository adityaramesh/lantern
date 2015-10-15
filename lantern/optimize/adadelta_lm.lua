--
-- A version of AdaDelta that uses less memory, but has the following disadvantages:
-- * Does not support logging. Logging slows things down and uses up extra
--   memory anyway, so if you need support for this, then use
--   `lantern.optimizers.adadelta`.
-- * May suffer from numerical issues in certain situations. If you suspect that
--   this is happening to you, try `lantern.optimizers.adadelta`.
--

local adadelta_lm = lantern.make_optimizer("adadelta_lm")

function adadelta_lm:__init(model, state)
	self.model       = model
	self.params      = model:parameters()
	self.grad_params = model:grad_parameters()
	self.state       = state or {}

	self.state.iter = self.state.iter          or 0
	self.eps        = self.state.eps           or 1e-10
	self.lr         = self.state.learning_rate or lantern.schedule.constant(1e-3)
	self.mom        = self.state.momentum      or lantern.schedule.constant(0.95)
	self.decay      = self.state.decay         or lantern.schedule.constant(0.95)
	self.mom_type   = self.state.momentum_type or lantern.momentum.none
end

function adadelta_lm:update(input, target)
	local iter = self.state.iter
	self.state.iter = self.state.iter + 1

	local cur_lr = self.lr(iter)
	local cur_decay = self.decay(iter)
	assert(cur_lr > 0 and cur_lr <= 1)
	assert(cur_decay > 0 and cur_decay < 1)

	-- Initializing the parameters here causes the first update to be
	-- multiplied by `(1 - cur_decay)`, since the running average of the
	-- second moment estimates will be zero. While it may seem like using a
	-- severe underestimate may impede convergence, I have actually found
	-- that the optimizer converges faster this way.
	if not self.state.temp then
		-- Used as a buffer to store intermediate values.
		self.state.temp = torch.Tensor():typeAs(self.params):
			resizeAs(self.params):zero()
		-- Estimate of the second moment of the gradient.
		self.state.grad_mom_2 = torch.Tensor():typeAs(self.params):
			resizeAs(self.params):zero()
		-- Estimate of the second moment of the update.
		self.state.update_mom_2 = torch.Tensor():typeAs(self.params):
			resizeAs(self.params):zero()
	end

	if self.mom_type == sopt.none then
		local outputs, loss = self.model:evaluate(input, target)
		self.state.temp:pow(self.grad_params, 2)
		self.state.grad_mom_2:mul(cur_decay):add(1 - cur_decay, self.state.temp)

		-- Note: adding and subtracting eps from the same quantity will
		-- not result in a contribution of zero in general. This may
		-- cause issues in certain situations.

		self.state.update_mom_2:add(self.eps)
		self.state.grad_mom_2:add(self.eps)
		self.state.temp:cdiv(self.state.update_mom_2, self.state.grad_mom_2):
			sqrt():cmul(self.grad_params):mul(-cur_lr)
		self.state.update_mom_2:add(-self.eps)
		self.state.grad_mom_2:add(-self.eps)
		self.params:add(self.state.temp)

		self.state.temp:pow(2):mul(1 - cur_decay)
		self.state.update_mom_2:mul(cur_decay):add(self.state.temp)

		return outputs, loss
	elseif self.mom_type == sopt.nag then
		if not self.state.step then
			self.state.step = torch.Tensor():typeAs(self.params):
				resizeAs(self.params):zero()
		end

		local cur_mom = self.mom(iter)
		assert(cur_mom > 0 and cur_mom < 1)

		-- Evaluate the function at the trial point.
		self.state.step:mul(cur_mom)
		self.params:add(self.state.step)
		local outputs, loss = self.model:evaluate(input, target)

		self.state.temp:pow(self.grad_params, 2)
		self.state.grad_mom_2:mul(cur_decay):add(1 - cur_decay, self.state.temp)

		-- Note: adding and subtracting eps from the same quantity will
		-- not result in a contribution of zero in general. This may
		-- cause issues in certain situations.

		self.state.update_mom_2:add(self.eps)
		self.state.grad_mom_2:add(self.eps)
		self.state.temp:cdiv(self.state.update_mom_2, self.state.grad_mom_2):
			sqrt():cmul(self.grad_params):mul(-cur_lr)
		self.state.update_mom_2:add(-self.eps)
		self.state.grad_mom_2:add(-self.eps)

		self.state.step:add(self.state.temp)
		self.params:add(self.state.temp)

		self.state.temp:pow(self.state.step, 2)
		self.state.update_mom_2:mul(cur_decay):add(1 - cur_decay, self.state.temp)

		return outputs, loss
	else
		error("Unsupported momentum type.")
	end
end
