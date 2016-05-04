local event = torch.class('lantern.event.event')

--[[
Requied parameters:
* `args.name`: Name of this event. No two events in the same `event_group` can have the same name.

Optional parameters:
* `args.update_period`: Period in iterations at which updates are produced (default is 1). If the
  current iteration is not divisible by `args.update_period, then `update` effectively does nothing.
  This parameter is useful for amortizing the space and time costs of events that generate large
  volumes of data each time `update` is called.
* `args.track`: If this event generates performance metrics (e.g. accuracy) and a `progress_tracker`
  is registered with it, then this parameter controls whether updates to these metrics are
  submitted to the `progress_tracker`. This is useful if, for instance, one wishes to monitor the
  value of a metric without having it influence the checkpointing process.
--]]
function event:__init(args)
	assert(type(args.name) == 'string')
	assert(lantern.is_valid_name(args.name), lantern.invalid_name_msg("Event name"))

	self.name          = args.name
	self.update_period = args.update_period or 1
	self.track         = args.track or true
end

--[[
Registers the given `event_group` with this `event`. The `event_group` provides all events owned by
it with the following information:
* The logger.
* The `progress_tracker`.
* The name of the directory to which the outputs of the `event_group` are written.

Note: this method should be oevrriden by the derived class in order to prepare the logger and
`progress_tracker` associated with the `event_group` to receive the information that will be
provided by this event.
--]]
function event:register_parent(group) end

--[[
If this event generates performance metrics, then this method can be used to control whether updated
values of these metrics are submitted to the `progress_tracker` of the parent `event_group`.
--]]
function event:track_progress(value)
	self.track = value
end

--[[
If the iteration is divisible by this event's update period, then the event is updated with the
information provided by `args`.

Note: this method should be overriden by the derived class in order to actually compute the update.
--]]
function event:update(args)
	assert(type(args.iteration) == 'number')
	assert(args.iteration >= 1)

	if args.iteration % self.update_period ~= 0 then return end
end

--[[
Generates and logs a summary of the information that this event has prcessed so far.

Note: this method should be overriden by the derived class if this functionality is to be supported
by this event.
--]]
function event:summarize() end
