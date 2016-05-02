local F = lantern.F
local event_group = torch.class('lantern.event.event_group')

function event_group:__init()
	self.events = {}
	self.name_to_event = {}

	--[[
	TODO:
	- initialize message pack logger
	- initialize progress tracker
	--]]
end

function event_group:add_event(e)
	assert(
		e:antecedent_count() == 0,
		"Continuations cannot be added to event groups."
	)
	assert(
		self.name_to_event[e.name] == nil,
		"An event with the name '{e.name}' already exists in the group."
	)

	table.insert(self.events, e)
	self.name_to_event[e.name] = e

	-- TODO set message pack logger
	-- TODO set progress tracker
end

--[[
Updates the event given by `name` with the information in the table `args`.
--]]
function event_group.update(name, args)
	self.name_to_event[name]:update(args)
end

--[[
Causes all registered events to generate and log summaries of the information that they have
processed so far.
--]]
function event_group:summarize()
	for _, e in pairs(self.events) do
		e:summarize()
	end
end

--[[
This function must be called each time an epoch is performed on the dataset corresponding to this
event group.
--]]
function event_group:reset()
	for _, e in pairs(self.events) do
		e:reset()
	end
end
