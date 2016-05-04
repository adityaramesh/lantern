local F = lantern.F
local pb = lantern.pb
local event_logger = torch.class('lantern.event_logger')

function event_logger:__init(args)
	assert(type(args.output_dir) == 'string')
	self.logger = args.logger

	--[[
	Since the event record definition is generated from scratch each time the `event_logger` is
	instantiated, we could parse the protobuf definition and start logging event data without
	actually writing the definition to disk. However, we still choose to do so for the following
	reasons:
	1. Writing out the definition in Torch format allows us to easily check that the previous
	   and current event record definitions are the same. This allows us to fail fast and alter
	   the user instead of silently logging event data in two different formats for the same
	   experiment.
	2. Writing out the definition in Protobuf format allows us to load the event data from other
	   languages, such as Python.
	--]]

	self.log_file_path    = paths.concat(args.output_dir, 'event_data.dat')
	self.t7_def_file_path = paths.concat(args.output_dir, 'event_record.t7')
	self.pb_def_file_path = paths.concat(args.output_dir, 'event_record.pb2')

	lantern.fail_if(paths.filep(log_file_path), F"The file '{self.log_file_path}' should " ..
		"have been renamed and given a suffix by the checkpointer. We cannot safely " ..
		"append to it, because hard links to it may exist in order directories.",
		self.logger)

	self.log_file = io.open(log_file_path, 'w')
	self.message_def = lantern.message_definition{name = 'EventRecord'}
	self.is_initialized = false

	self.message_def:add_field(lantern.field_definition{
		name = "epoch", type = "uint32", required = true})
	self.message_def:add_field(lantern.field_definition{
		name = "iteration", type = "uint32", required = true})
end

function event_logger:close()
	self.log_file:close()
end

function event_logger:add_event(event_def)
	assert(not self.is_initialized, "Events cannot be added to the event logger after it " ..
		"has already been initialized.")
	self.message_def:add_message(event_def)
end

function event_logger:initialize()
	if paths.filep(self.t7_def_file_path) then
		local old_def = torch.load(self.t7_def_file_path)
		lantern.fail_if(self.message_def ~= old_def,
			F"Old event record definition file {self.t7_def_file_path} differs from " ..
			"the current definition. Continuing would result in event data in two " ..
			"different formats for the same experiment.", self.logger)
	else
		torch.save(self.t7_def_file_path, self.message_def)
	end

	local proto = self.message_def:to_proto()
	local f = io.open(self.pb_def_file_path, 'w')
	f:write(proto)
	f:close()

	self.msg = db.load_proto(proto)
	self.is_initialized = true
end

function event_logger:register_checkpointer(c)
	-- TODO
end

--[[
Overriden so that users can access the fields of the current event record corresponding to the
events that have been added to this `event_logger`.
--]]
function event_logger:__index(k)
	if self.msg ~= nil and self.msg.k ~= nil then
		return self.msg.k
	end

	return getmetatable(self)[k]
end

--[[
Overriden so that users can easily set the epoch and iteration values of the current event record.
--]]
function event_logger:__newindex(k, v)
	if self.msg ~= nil and self.msg.k ~= nil then
		self.msg.k = v
		return
	end

	rawset(self, k, v)
end

function event_logger:flush()
	self.output_file:write(msg:Serialize())
	msg:Clear()
end
