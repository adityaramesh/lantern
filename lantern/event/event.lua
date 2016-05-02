local event = torch.class('lantern.event.event')

function event:__init(args)
	assert(args.name ~= nil)
	assert(type(args.name) == 'string')

	assert(args.is_numeric ~= nil)
	assert(type(args.is_numeric) == 'boolean')

	assert(args.is_totally_ordered ~= nil)
	assert(type(args.is_totally_ordered) == 'boolean')

	assert(args.update_period ~= nil)
	assert(type(args.update_period) == 'number')
	assert(args.update_period >= 1)

	self.name               = args.name
	self.is_muted           = args.mute or false
	self.is_totally_ordered = args.is_totally_ordered
	self.update_period      = args.update_period

	self.antecedents   = {}
	self.continuations = {}

	self.update_deps       = {}
	self.summary_deps      = {}
	self.update_dep_count  = 0
	self.summary_dep_count = 0
end

function event:add_continuation(n)
	assert(
		n.update_period == 1,
		"Any continuation must have an update period of one, since its update period is " ..
		"effectively the LCM of those of its parents."
	)

	table.insert(self.continuations, n)
	table.insert(n.antecedents, self)
	table.insert(n.dependencies, false)
	n.dep_count = n.dep_count + 1
end

function event:antecedent_count()
	return #self.antecedents
end

function event:continuation_count()
	return #self.continuations
end

function event:is_continuation()
	return self:antecedent_count() >= 1
end

--[[
While muted, an event will no longer log events or make updates to its progress tracker, if it is
registered with one. Note that muting an event does not mute its continuations.
--]]
function event:mute()
	self.is_muted = true
end

function event:unmute()
	self.is_muted = false
end

--[[
Sets the logger for this event, and prepares it to receive the information that will be provided by
this event.
--]]
function event:set_logger(logger)
	assert(self.is_numeric, F"Event {self.name} is non-numeric and hence cannot be logged.")
	self.logger = logger

	for _, e in pairs(self.continuations) do
		if e.logger == nil then e:set_logger(logger) end
	end

	self:set_logger_impl(logger)
end

--[[
Sets the progress tracker for this event, and prepares it to track the information that will be
provided by this event.
--]]
function event:set_progress_tracker(tracker)
	assert(
		self.is_totally_ordered,
		F"Event '{self.name}' is not totally ordered and hence cannot be tracked."
	)
	self.tracker = tracker

	for _, e in pairs(self.continuations) do
		if e.tracker == nil then e:set_progress_tracker(tracker) end
	end

	self:set_progress_tracker_impl(tracker)
end

function event:reset_update_dependencies()
	self.update_dep_count = 0

	for i in 1, #self.update_deps do
		self.update_deps[i] = false
	end
end

function event:reset_summary_dependencies()
	self.summary_dep_count = 0

	for i in 1, #self.summary_deps do
		self.summary_deps[i] = false
	end
end

function event:reset()
	self:reset_update_dependencies()
	self:reset_summary_dependencies()

	for _, e in pairs(self.continuations) do
		e:reset()
	end
end

--[[
If the iteration is divisible by this event's update period, then the event is updated with the
provided information, and `update` is invoked on each continuation. A continuation is only updated
when all parents have invoked `update` on it at least once. After this happens, the continuation is
updated, and the counters are reset.
--]]
function event:update(args)
	if not self:is_continuation() then
		assert(args.iteration ~= nil)
		assert(args.iteration >= 1)

		if args.iteration % self.update_period ~= 0 then return end
		self:update_impl(args)

		for i, e in pairs(self.continuations) do
			e:update({parent = i})
		end
	else
		assert(self.update_period == 1)
		assert(args.parent ~= nil)
		assert(args.parent >= 1)

		local count = self:antecedent_count()
		assert(count == #self.update_deps)
		assert(count == #self.summary_deps)

		if not self.update_deps[args.parent] then
			self.update_dep_count = self.update_dep_count + 1
			self.update_deps[args.parent] = true
		end

		if self.update_dep_count ~= count then return end
		self:reset_update_dependencies()
		self:update_impl(args)

		for i, e in pairs(self.continuations) do
			e:update({parent = i})
		end
	end
end

--[[
Generates and logs a summary of this event. If this event has continuations, then `summarize` is
invoked on each continuation. A continuation only generates a summary when each parent has called
`summarize` on it once. At this point, the summary is generated, and the counters are reset. If a
continuation has more than one parent, then it is an error for the the same parent to call
`summarize` on the continuation twice in succession.
--]]
function event:summarize(args)
	if not self:is_continuation() then
		assert(args == nil)
		self:summarize_impl()

		for i, e in pairs(self.continuations) do
			e:summarize({parent = i})
		end
	else
		assert(self.update_period == 1)
		assert(args.parent ~= nil)
		assert(args.parent >= 1)

		local count = self:antecedent_count()
		assert(count == #self.update_deps)
		assert(count == #self.summary_deps)

		assert(
			self.summary_deps[args.parent] == false,
			"If a continuation has more than one parent, then the same parent cannot " ..
			"call 'summarize' on it twice in succession."
		)

		self.summary_dep_count = self.summary_dep_count + 1
		self.summary_deps[args.parent] = true

		if self.summary_dep_count ~= count then return end
		self:reset_summary_dependencies()
		self:summarize_impl(args)

		for i, e in pairs(self.continuations) do
			e:summarize({parent = i})
		end
	end
end

--[[
The derived class must implement the functions below that should not be no-ops.
--]]
function event:set_logger_impl(logger) end
function event:set_progress_tracker_impl(tracker) end
function event:update_impl(args) end
function event:summarize_impl() end
