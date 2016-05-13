local F = lt.F
local event_metrics = torch.class('lt.event_metrics')
local progress_tracker, parent = torch.class('lt.progress_tracker', 'lt.serializable')

function event_metrics:__init(metrics)
	self.metric_info = {}

	for name, func in pairs(metrics) do
		self.metric_info[name] = {
			improved = false,
			has_improved = func,
		}
	end
end

function event_metrics:__index__(k)
	local metric_info = rawget(self, 'metric_info')

	if metric_info ~= nil and metric_info[k] ~= nil then
		return metric_info[k].value, true
	end

	return false
end

function event_metrics:__newindex__(k, v)
	if self.metric_info ~= nil and self.metric_info[k] ~= nil then
		local info = self.metric_info[k]

		if info.value == nil or info.has_improved(info.value, v) then
			info.value = v
			info.improved = true
		end

		return true
	end

	return false
end

function event_metrics:improved_metrics()
	local improved = nil

	for metric, info in pairs(self.metric_info) do
		if info.improved then
			improved = improved or {}
			table.insert(improved, metric)
			info.improved = false
		end
	end

	return improved
end

function progress_tracker:__init(args)
	self.state = {
		--
		logger = args.logger,
		event_info = {}
	}
end

--[[
Parameters:
* `name`: The name of an event that generates one or more performance metrics.
* `metrics`: A table describing the metrics associated with event `name` that are to be tracked.
  Each key in the table is the name of an metric, and each value is a binary function used to
  determine whether a new value of the metric (the second argument to the function) is an
  improvement over the old one (the first argument to the function).
--]]
function progress_tracker:add_event(name, metrics)
	if self.state.event_info[name] ~= nil then return end
	self.state.event_info[name] = lt.event_metrics(metrics)
end

function progress_tracker:improved_metrics()
	local improved = {}

	for event, info in pairs(self.state.event_info) do
		local list = info:improved_metrics()
		if list ~= nil then improved[event] = list end
	end

	return improved
end

function progress_tracker:__index__(k)
	local state = rawget(self, 'state')
	if state == nil then return false end
	local event_info = state.event_info

	if event_info ~= nil and event_info[k] ~= nil then
		return event_info[k], true
	end

	return false
end
