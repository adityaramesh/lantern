local event = torch.class('lantern.event.event')

function event:__init(args)
	assert(type(args.name) == 'string')
	assert(type(args.is_totally_ordered) == 'boolean')
	assert(type(args.update_period) == 'number')
	assert(args.update_period >= 1)

	self.name               = args.name
	self.is_totally_ordered = args.is_totally_ordered
	self.update_period      = args.update_period
end

--[[
Sets the logger for this event, and prepares it to receive the information that will be provided by
this event.
--]]
function event:set_logger(logger) end

--[[
Sets the progress tracker for this event, and prepares it to track the information that will be
provided by this event.
--]]
function event:set_progress_tracker(tracker) end

--[[
If the iteration is divisible by this event's update period, then the event is updated with the
information provided by `args`.
--]]
function event:update(args)
	assert(type(args.iteration) == 'number')
	assert(args.iteration >= 1)

	if args.iteration % self.update_period ~= 0 then return end
	self:update_impl(args)
end

function event:update_impl(args) end

--[[
Generates and logs a summary of the information that this event has prcessed so far.
--]]
function event:summarize() end
