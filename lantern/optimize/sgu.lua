local sgu = lantern.make_optimizer("sgu")

--
-- Note: the `model` parameter here is unused, but kept anyway to preserve API
-- uniformity. Other optimization algorithms may need to use this parameter to
-- perform model-specific operations (e.g. disabling/enabling dropout).
--
function sgu:__init(model, state, logger)
	self.model       = model
	self.params      = model:parameters()
	self.grad_params = model:grad_parameters()
	self.state       = state or {}

	self.state.name = "sgu"
	self.state.iter = self.state.iter          or 0
	self.lr         = self.state.learning_rate or lantern.schedule.constant(1e-3)
	self.mom        = self.state.momentum      or lantern.schedule.constant(0.95)
	self.mom_type   = self.state.momentum_type or lantern.momentum.none

	if logger then
		self.logger = logger
		self.logger:add_fields({"loss", "norm_grad", "theta", "eta_a", "eta_w"})
	end
end

--
-- Log function for SGU without NAG.
--
function sgu:log_info(batch, cur_lr, loss)
	if not self.logger then return end

	if not self.prev_grad_params then
		self.prev_grad_params = torch.Tensor():typeAs(self.grad_params):
			resizeAs(self.grad_params)
	end

	local norm_grad = self.grad_params:norm()
	-- In our case, descent := p_k g_k = -||g_k||^2.
	local descent = -math.pow(norm_grad, 2)
	self.prev_grad_params:copy(self.grad_params)
	local new_loss = self.model:evaluate(batch)

	local eta_a = (new_loss -  loss) / (cur_lr * descent)
	local eta_w = math.abs(-self.grad_params:dot(
		self.prev_grad_params) / descent)

	self.logger:log_value("loss", loss)
	self.logger:log_value("norm_grad", norm_grad)
	self.logger:log_value("theta", math.pi)
	self.logger:log_value("eta_a", eta_a)
	self.logger:log_value("eta_w", eta_w)
end

--
-- Log function for SGU with NAG.
--
-- Precondition: `self.grad_params` is *not* modified.
-- Precondition: `self.state.step` =: s_k is defined such that
-- `x_{k + 1} = x_k + s_k`.
--
function sgu:log_nag_info(batch, cur_lr, loss)
	if not self.logger then return end
	assert(self.state.prev_params ~= nil)
	assert(self.state.prev_grad_params ~= nil)

	-- In order to allow for a more direct comparison to SGD, we
	-- make the following observation regarding the NAG update:
	-- 	s_{k + 1} :=  mu * s_k - eta * hat{g}_{k + 1}
	-- 	           =  eta(mu / eta * s_k - hat{g}_{k + 1})
	-- 	           =: eta * p_{k + 1}.
	-- So the analogous notion of "search direction" for SGU with NAG is
	-- p_{k + 1} = (1 / eta) * s_{k + 1}. Thus we use p_{k + 1} instead of
	-- s_{k + 1} to compute the quantities below.

	local norm_grad = self.state.prev_grad_params:norm()
	-- Note that descent := `1 / cur_lr * proj`. Because of cancellation
	-- with `cur_lr` that occurs in the formulas, we don't actually define
	-- it this way.
	local proj = self.state.step:dot(self.state.prev_grad_params)
	-- Note that theta could be NaN. If this happens, then either the update
	-- or the gradient has very small magnitude, so the angle could not be
	-- computed in single precision.
	local theta = math.acos(proj / (self.state.step:norm() * norm_grad))

	local new_loss = self.model:evaluate(batch)
	local eta_a = cur_lr * (new_loss - loss) / proj
	local eta_w = math.abs(self.state.step:dot(self.grad_params) / proj)

	self.logger:log_value("loss", loss)
	self.logger:log_value("norm_grad", norm_grad)
	self.logger:log_value("theta", theta)
	self.logger:log_value("eta_a", eta_a)
	self.logger:log_value("eta_w", eta_w)
end

function sgu:update(batch)
	local iter = self.state.iter
	self.state.iter = self.state.iter + 1

	local cur_lr = self.lr(iter)
	assert(cur_lr > 0 and cur_lr <= 1)

	if self.mom_type == lantern.momentum.none then
		local state = self.model:evaluate(batch)
		self.params:add(-cur_lr, self.grad_params)
		self:log_info(batch, cur_lr, loss)
		return state
	elseif self.mom_type == lantern.momentum.nag then
		-- For the first iteration, we just take the direction of
		-- steepest descent.
		if not self.state.step then
			local state = self.model:evaluate(batch)
			self.state.step = self.grad_params:clone():mul(-cur_lr)
			self.params:add(self.state.step)

			if self.logger then
				-- Unlike vanilla SGU, `prev_params` and
				-- `grad_params` need to be part of the state,
				-- since the logging function depends on their
				-- values and does not just use them as
				-- temporary buffers.
				self.state.prev_params = self.params:clone()
				self.state.prev_grad_params = self.grad_params:clone()
				self:log_nag_info(batch, cur_lr, loss)
			end

			return state
		end

		local cur_mom = self.mom(iter)
		assert(cur_mom > 0 and cur_mom < 1)

		-- Evaluate the function at the trial point.
		self.state.step:mul(cur_mom)
		self.params:add(self.state.step)
		local state = self.model:evaluate(batch)

		-- Update the parameters. We don't multiply the gradient by
		-- `-cur_lr` in advance because the logging function requires
		-- the original value.
		self.state.step:add(-cur_lr, self.grad_params)
		self.params:add(-cur_lr, self.grad_params)

		if self.logger then
			self:log_nag_info(batch, cur_lr, loss)
			self.state.prev_params:copy(self.params)
			self.model:evaluate(batch)
			self.state.prev_grad_params:copy(self.grad_params)
		end

		return state
	else
		error("Unsupported momentum type.")
	end
end
