--
-- Utility functions for measuring improvements in the training and testing
-- metrics.
-- 

function lantern.improved_metrics(old, new)
	local improved = {}

	for k, v in pairs(old) do
		local dir = lantern.performance_metrics[k]

		if not new[k] and dir then
			improved[k] = v
		elseif new[k] and dir == "increasing" then
			if v > old[k] then improved[k] = v end
		elseif new[k] and dir == "decreasing" then
			if v < old[k] then improved[k] = v end
		end
	end

	return improved
end

function lantern.improvement_made(old, new)
	local improved = lantern.improved_metrics(old, new)
	local count = 0

	for k, v in pairs(improved) do
		count = count + 1
	end

	return count
end

function lantern.best_metrics(hist, mode, epochs)
	epochs = epochs or #hist
	assert(epochs >= 1)
	assert(epochs <= #hist)
	assert(mode == "train" or mode == "test")

	local update_metrics = function(prev_best, cur)
		for k, v in pairs(cur) do
			-- If `dir` is not defined for a metric, then we assume
			-- that it does not make sense to measure whether it has
			-- improved or not. So we avoid tracking it.
			local dir = lantern.performance_metrics[k]

			if not prev_best[k] and dir then
				prev_best[k] = v
			elseif prev_best[k] and dir == "increasing" then
				if v > prev_best[k] then prev_best[k] = v end
			elseif prev_best[k] and dir == "decreasing" then
				if v < prev_best[k] then prev_best[k] = v end
			end
		end
	end

	local best_metrics = {}
	for i = 1, epochs do
		if hist[i][mode] then
			update_metrics(best_metrics, hist[i][mode])
		end
	end

	return best_metrics
end
