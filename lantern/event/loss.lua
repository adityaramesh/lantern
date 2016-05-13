local F = lt.F
local loss, parent = torch.class('lt.loss', 'lt.event')

function loss:__init(args)
	parent.__init(self, args)
	self:initialize()
end

function loss:initialize()
	self.count = 0
	self.total = 0
end

function loss:register_parent(group)
	assert(self.state.parent == nil, F"A parent group called '{self.state.parent:name()}'" ..
		"has already been registered with this event.")

	self.state.parent = group

	local fields = {
		lt.field_definition{name = 'value', type = 'double'},
		lt.field_definition{name = 'mean', type = 'double'},
	}

	group.event_logger:add_event(self:name(), fields)

	if self.state.track then
		group.tracker:add_event(self:name(),
			{mean = function (old, new) return new < old end})
	end
end

function loss:update(args)
	if not parent.update(self, args) then return end

	assert(type(args.value) == 'number')
	self.state.parent.event_logger[self:name()].value = args.value

	self.count = self.count + 1
	self.total = self.total + args.value
end

function loss:summarize(args)
	assert(self.count ~= 0)

	local mean = self.total / self.count
	self.state.parent.event_logger[self:name()].mean = mean
	if self.state.track then self.state.parent.tracker[self:name()].mean = mean end

	self.total = 0
	self.count = 0
end
