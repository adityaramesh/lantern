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

	return function(hist)
		assert(hist[#hist].train or hist[#hist].test)

		-- If we want to check for improvement over the past one epoch,
		-- then we need at least two entries. So if we have fewer than
		-- `epochs + 1` entries, then we return `true` since we have
		-- insufficient data.
		if #hist <= epochs then return true end

		local best_metrics = {
			train = lantern.best_metrics(hist, "train", #hist - epochs),
			test = lantern.best_metrics(hist, "test", #hist - epochs)
		}

		local size = function(table)
                        local count = 0
                        for k, v in pairs(table) do count = count + 1 end
                        return count
                end

                if size(best_metrics.train) == 0 then best_metrics.train = nil end
                if size(best_metrics.test) == 0 then best_metrics.test = nil end

		for i = #hist - epochs + 1, #hist do
			if best_metrics.train and hist[i].train then
				if lantern.improvement_made(best_metrics.train, hist[i].train) then
					return true
				end
			end

			if best_metrics.test and hist[i].test then
				if lantern.improvement_made(best_metrics.test, hist[i].test) then
					return true
				end
			end
		end

		return false
	end
end

--
-- Returns true if an improvement has been made over the best value of a *test*
-- metric before the past `epochs` epochs during the past `epochs` epochs. This
-- stopping criterion is designed to be conservative, and may not be appropriate
-- when training large numbers of models.
--
function lantern.criterion.max_epochs_per_improvement(epochs)
	assert(epochs > 0)

	return function(hist)
		assert(hist[#hist].test)

		-- If we want to check for improvement over the past one epoch,
		-- then we need at least two entries. So if we have fewer than
		-- `epochs + 1` entries, then we return `true` since we have
		-- insufficient data.
		if #hist <= epochs then return true end

		local best_metrics = lantern.best_metrics(hist, "test", #hist - epochs)

		local size = function(table)
                        local count = 0
                        for k, v in pairs(table) do count = count + 1 end
                        return count
                end

                assert(size(best_metrics) >= 1)

		for i = #hist - epochs + 1, #hist do
			if hist[i].test then
				if lantern.improvement_made(best_metrics, hist[i].test) then
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
