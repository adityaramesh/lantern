--
-- Definitions of the stopping criteria.
--

--
-- Returns true if an improvement has been made for any metric during training
-- or validation over the past number of epochs given by `epochs`. This stopping
-- criterion is designed to be conservative, and may not be appropriate when
-- training large numbers of models.
--
function lantern.criterion.max_epochs_per_improvement(epochs)
	assert(epochs > 0)

	local function improvement_made(old, new)
		for k, v in new do
			if lantern.performance_metrics[k] == "increasing" then
				if v > old[k] then
					return true
				end
			elseif v < old[k] then
				return true
			end
		end
		return false
	end

	return function(state)
		-- If we want to check for improvement over the past one epoch,
		-- then we need at least two entries. So if we have fewer than
		-- `epochs + 1` entries, then we return `true` since we have
		-- insufficient data.
		if #state <= epochs then return true end

		for i = #state, #state - epochs + 1, -1 do
			for j = i - 1, #state - epochs, -1 do
				if state[j].train and state[i].train then
					if improvement_made(state[j].train, state[i].train) then
						return true
					end
				end

				if state[j].test and state[i].test then
					if improvement_made(state[j].test, state[i].test) then
						return true
					end
				end
			end
		end

		return false
	end
end
