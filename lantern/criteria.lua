--
-- Definitions of the stopping criteria.
--

--
-- Returns true if an improvement has been made over the best value of a metric
-- before the past `epochs` epochs during the past `epochs` epochs. This
-- stopping criterion is designed to be conservative, and may not be appropriate
-- when training large numbers of models.
--
function lantern.criterion.max_epochs_per_improvement(epochs)
	assert(epochs > 0)

	local function improvement_made(old, new)
		for k, v in pairs(new) do
			if lantern.performance_metrics[k] == "increasing" then
				if v > old[k] then return true end
			elseif lantern.performance_metrics[k] == "decreasing" then
				if v < old[k] then return true end
			end
		end
		return false
	end

	return function(hist)
		assert(hist[#hist].train or hist[#hist].test)

		-- If we want to check for improvement over the past one epoch,
		-- then we need at least two entries. So if we have fewer than
		-- `epochs + 1` entries, then we return `true` since we have
		-- insufficient data.
		if #hist <= epochs then return true end

		local update_metrics = function(best, cur)
			for k, v in pairs(cur) do
				local dir = lantern.performance_metrics[k]

				if not best[k] and dir then
					best[k] = v
					break
				end

				if dir == "increasing" then
					if v > best[k] then
						best[k] = v
					end
				elseif dir == "decreasing" then
					if v < best[k] then
						best[k] = v
					end
				end
			end
		end

		local best_metrics = {}

		for i = 1, #hist - epochs do
			if hist[i].train then
				best_metrics.train = best_metrics.train or {}
				update_metrics(best_metrics.train, hist[i].train)
			end

			if hist[i].test then
				best_metrics.test = best_metrics.test or {}
				update_metrics(best_metrics.test, hist[i].test)
			end
		end

		for i = #hist - epochs + 1, #hist do
			if best_metrics.train and hist[i].train then
				if improvement_made(best_metrics.train, hist[i].train) then
					return true
				end
			end

			if best_metrics.test and hist[i].test then
				if improvement_made(best_metrics.test, hist[i].test) then
					return true
				end
			end
		end

		return false
	end
end

function lantern.criterion.max_epochs(epochs)
	assert(epochs > 0)

	return function(hist)
		assert(hist[#hist].train or hist[#hist].test)
		if #hist <= epochs - 1 then return true end
		return false
	end
end
