local loss, parent = torch.class('lt.loss', 'lt.event')

function loss:__init(args)
	parent.__init(self, args)
	self.count = 0
	self.total = 0
end

function loss:register_parent(group)
	self.logger = group.logger
	self.tracker = group.tracker

	local fields = {
		lt.field_definition{name = 'value', type = 'double'},
		lt.field_definition{name = 'mean', type = 'double'},
	}
	self.logger:add_event(self.name, fields)

	if self.track then
		self.tracker:add_event(self.name, {mean = function (old, new) return new < old end})
	end
end

function loss:update(args)
	parent.update(self, args)

	assert(type(args.value) == 'number')
	self.logger[self.name].value = args.value

	self.count = self.count + 1
	self.total = self.total + args.value
end

function loss:summarize(args)
	assert(self.count ~= 0)

	local mean = self.total / self.count
	self.logger[self.name].mean = mean
	if self.track then self.tracker[self.name].mean = mean end

	self.total = 0
	self.count = 0
end
