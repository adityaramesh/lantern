local F = lt.F
local event_metrics = torch.class('lt.event_metrics')
local progress_tracker = torch.class('lt.progress_tracker')

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
	assert(type(args.output_dir) == 'string')

	self.logger = args.logger
	self.output_file_path = paths.concat(args.output_dir, 'event_metrics.t7')
	self.is_initialized = false

	if paths.filep(self.output_file_path) then
		if self.logger ~= nil then
			self.logger:log('/console/info', "Loading previous event metric data " ..
				"from '{self.output_file_path}'.")
		end

		self.event_info = torch.load(self.output_file_path)
	else
		self.event_info = {}
	end
end

function progress_tracker:register_checkpointer(c)
	-- TODO
end

--[[
Parameters:
* `name`: The name of the event.
* `metrics`: A table describing the metrics associated with the event that are to be tracked. Each
  key in the table is the name of an event, and each value is a binary function used to determine
  whether a new value of the metric (the second argument) is an improvement over the old one (the
  first argument).
--]]
function progress_tracker:add_event(name, metrics)
	assert(not self.is_initialized, "Events cannot be added to this progress tracker after " ..
		"it has already been initialized.")

	if self.event_info[name] ~= nil then return end
	self.event_info[name] = lt.event_metrics(metrics)
end

function progress_tracker:initialize()
	assert(not self.is_initialized, "Event group has already been initialized.")
	self.is_initialized = true
end

function progress_tracker:improved_metrics()
	local improved = {}

	for event, info in pairs(self.event_info) do
		local list = info:improved_metrics()
		if list ~= nil then improved[event] = list end
	end

	return improved
end

function progress_tracker:__index__(k)
	local event_info = rawget(self, 'event_info')

	if event_info ~= nil and event_info[k] ~= nil then
		return event_info[k], true
	end

	return false
end

function progress_tracker:flush()
	torch.save(self.output_file_path, self.event_info)
end
