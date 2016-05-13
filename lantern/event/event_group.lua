local F = lt.F
local event_group = torch.class('lt.event_group', 'flow.serializable')

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
	assert(lt.is_name_valid(args.name), lt.invalid_name_msg("Event group name"))

	self.state = {
		name          = args.name,
		logger        = args.logger,
		name_to_event = {}
	}

	self:initialize()
end

function event_group:name()
	return self.name
end

--[[
Adds the given event to this `event_group`.
--]]
function event_group:add_event(e)
	assert(not self.is_initialized, "Events cannot be added to this event group after it " ..
		"has already been initialized.")
	assert(self.state.name_to_event[e.name] == nil, F"Event with name '{e.name}' is already " ..
		"added to this group.")
	self.state.name_to_event[e:name()] = e
end

--[[
Parameters:
* `args.cur_ver_dir`: Path to directory intended to contain the results of the current version of
  the experiment (e.g. the `current` subdirectory within the root directory for this experiment).

**Note:** if the description for `args.cur_ver_dir` is updated, also update the one in
`lantern/event/checkpointer.lua`.
--]]
function event_group:initialize(args)
	--[[
	This branch is taken in two situations:
	1. A new `event_group` has been constructed.
	2. An `event_group` has been deserialized.

	The `initialize` function still needs to be called again, with the required parameters
	provided in `args`.
	--]]
	if args == nil then
		self.is_initialized = false
		return
	end

	assert(not self.is_initialized, "Event group has already been initialized.")
	assert(paths.dirp(args.cur_ver_dir), "Directory '{args.cur_ver_dir}' for the current " ..
		"version of the experiment does not exist.")

	self.output_dir = paths.concat(args.cur_ver_dir, self.state.name)
	lt.make_directory_if_not_exists(self.output_dir, self.state.logger)

	local output_args = {output_dir = self.output_dir, logger = self.state.logger}
	self.event_logger = lt.event_logger(output_args)
	self.tracker      = lt.progress_tracker(output_args)

	for _, e in pairs(self.state.name_to_event) do
		e:register_parent(self)
	end

	self.event_logger:initialize()
	self.is_initialized = true
end

--[[
Registers the given `checkpointer` with this `event_group`.
--]]
function event_group:register_checkpointer(c)
	self.event_logger:register_checkpointer(c)
	self.tracker:register_checkpointer(c)
end

--[[
Overriden so that users can access `event`s owned by this `event_group` in order to inspect or
update them.
--]]
function event_group:__index__(k)
	local state = rawget(self, 'state')
	if state == nil then return false end
	local name_to_event = state.name_to_event

	if name_to_event ~= nil and name_to_event[k] ~= nil then
		return name_to_event[k], true
	end

	return false
end

--[[
Flushes all updates that have been made logged to disk. If this `event_group` is associated with a
data set, then this function should be probably called after each iteration.
--]]
function event_group:flush_updates()
	self.event_logger:flush()
end

--[[
Invokes `summarize` on all `event`s added to this `event_group`, and returns the list of metrics
that have improved since the last call to `summarize`. If this `event_group` is associated with a
data set, then this function should probably be called after each epoch.
--]]
function event_group:summarize()
	for _, e in pairs(self.state.name_to_event) do
		e:summarize()
	end

	self.tracker:flush()
	return self.tracker:improved_metrics()
end
