--
-- Defines the `data_source` concept, and provides some implementations.
--

local data_source                = lantern.make_class('data_source')
local classification_data_source = lantern.make_class('classification_data_source', data_source)

function classification_data_source:__init(args)
	-- The name is required for **all** data sources, because the serializer uses them when
	-- creating and updating checkpoints.
	assert(args.name)
	assert(type(args.name) == 'string')
	self.name = args.name

	assert(args.class_count)
	assert(type(args.class_count) == 'number')
	self.class_count = args.class_count

	assert(args.inputs)
	assert(args.targets)

	if args.inputs._dataspaceID and args.targets._dataspaceID then
		-- We are reading from a HDF5 file.
		self.inputs = args.inputs:all()
		self.targets = agrs.targets:all()
	elseif args.inputs.__typename and args.targets.__typename then
		-- The arguments are `Tensor`s.
		self.inputs = args.inputs
		self.targets = args.targets
	else
		error("At least one of 'inputs' or 'targets' is invalid.")
	end
	
	assert(self.inputs:size(1) == self.targets:size(1))
	assert(self.class_count > 0)
	self.size = self.inputs:size(1)

	local access_stategy_factory = args.access_strategy or sequential_access_strategy()
end
