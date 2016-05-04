local event_group = torch.class('lantern.event.event_group')

--[[
Required parameters:
* `args.name`: Name of this event group. Most often, this is the name of the dataset associated with
   the events that we would like to monitor.

Optional parameters:
* `args.logger`: If provided, used to log auxiliary information, such as IO errors. This is **not**
  the same logger used by this group to log the high-frequency information produced by the managed
  `event`s.
--]]
function event_group:__init(args)
	assert(type(args.name) == 'string')
	assert(lantern.is_valid_name(args.name), lantern.invalid_name_msg("Event group name"))

	self.name   = args.name
	self.logger = args.logger

	self.events        = {}
	self.name_to_event = {}
	self.initialized   = false
end

--[[
Parameters:
* `prev_experiment_dir`: Directory containing the results of the previous experiments from which
  we are continuing (should be `nil` if we are starting from scratch).
* `cur_experiment_dir`: Directory to contain the results of the current experiment (this could
  potentially be the same as `prev_experiment_dir`).
--]]
function event_group:initialize(args)
	assert(not self.initialized, "Event group has already been initialized.")
	assert(paths.dirp(args.cur_experiment_dir), "Directory for current experiment does not " ..
		"exist.")

	self.output_dir = paths.concat(args.cur_experiment_dir, self.name)
	lantern.make_directory_if_not_exists(self.output_dir, self.logger)

	local args = {output_dir = self.output_dir, logger = self.logger}
	self.logger  = lantern.event.logger(args)
	self.tracker = lantern.event.progress_tracker(args)

	for _, e in pairs(self.events) do
		e:register_parent(self)
	end
end

--[[
Adds the given event to this `event_group`.
--]]
function event_group:add_event(e)
	assert(not self.initialized, "Events cannot be added to this event group after it has " ..
		"already been initialized.")
	assert(self.name_to_event[e.name] == nil, F"Event with name '{e.name}' is already " ..
		"added to this group.")

	table.insert(self.events, e)
	self.name_to_event[e.name] = e
end

--[[
Registers the given `checkpointer` with this `event_group`.
--]]
function event_group:register_checkpointer(c)
	self.logger:register_checkpointer(c)
	self.tracker:register_checkpointer(c)
end

--[[
Invokes `update` on the event with the given name.
--]]
function event_group:update(name, args)
	assert(self.initialized, "Event group has not yet been initialized.")
	self.name_to_event[name]:update(args)
end

--[[
Invokes `summarize` on all `event`s added to this `event_group`.
--]]
function event_group:summarize()
	assert(self.initialized, "Event group has not yet been initialized.")

	for _, e in pairs(self.events) do
		e:summarize()
	end
end

--[[
Flushes all updates that have been made logged to disk. If this `event_group` is associated with a
data set, then this function should be probably called after each iteration.
--]]
function event_group:flush_updates()
	self.logger:flush()
end

--[[
Returns the list of metrics that have improved since the last call to `improved_metrics`, and
flushes the latest table of the best recorded metrics to disk.
--]]
function event_group:improved_metrics()
	self.tracker:flush()
	return self.tracker:improved_metrics()
end
