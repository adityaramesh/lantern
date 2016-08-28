local optimizer_base, parent = torch.class('lt.optimizer_base', 'lt.serializable')

function optimizer_base:__init(args)
	assert(type(args.tensor_type) == 'string')
	assert(torch.type(args.tensor_size) == 'torch.LongStorage')

	parent.__init(self)
	self.state.tensor_type = args.tensor_type
	self.state.tensor_size = args.tensor_size
	self.state.logger      = args.logger
end

function optimizer_base:initialize()
	self.tensor_factory = torch.factory(self.state.tensor_type)
end

function optimizer_base:__load(args)
	parent.__load(self, args)
	self:initialize()
end
