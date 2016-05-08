local F = lt.F
local pb = lt.pb
local event_logger = torch.class('lt.event_logger')

function event_logger:__init(args)
	assert(type(args.output_dir) == 'string')
	self.logger = args.logger

	--[[
	Since the event record definition is generated from scratch each time the `event_logger` is
	instantiated, we could parse the protobuf definition and start logging event data without
	actually writing the definition to disk. However, we still choose to do so for the following
	reasons:
	1. Writing out the definition in Torch format allows us to easily check that the previous
	   and current event record definitions are the same. This allows us to fail fast and alert
	   the user instead of silently logging event data in two different formats for the same
	   experiment.
	2. Writing out the definition in Protobuf format allows us to load the event data from other
	   languages, such as Python.
	--]]

	self.is_initialized   = false
	self.log_file_path    = paths.concat(args.output_dir, 'event_data.dat')
	self.t7_def_file_path = paths.concat(args.output_dir, 'event_record.t7')
	self.pb_def_file_path = paths.concat(args.output_dir, 'event_record.pb2')

	lt.fail_if(paths.filep(self.log_file_path), F"The file '{self.log_file_path}' should " ..
		"have been renamed and given a suffix by the checkpointer. We cannot safely " ..
		"append to it, because hard links to it may exist in order directories.",
		self.logger)

	self.message_def = lt.message_definition{def_name = 'Wrapper'}
	self.record_def  = lt.message_definition{def_name = 'EventRecord', name = 'record',
		rule = 'required'}

	self.message_def:add_message(self.record_def)
	self.required_fields = {epoch = true, iteration = true}

	for f, _ in pairs(self.required_fields) do
		self.record_def:add_field(lt.field_definition{
			name = f, type = 'uint32', required = true})
	end

	self.event_to_def = {}
end

function event_logger:close()
	self.log_file:close()
end

function event_logger:add_event(name, fields)
	assert(not self.is_initialized, "Events cannot be added to the event logger after it " ..
		"has already been initialized.")
	assert(self.event_to_def[name] == nil, "Event with name '{name}' has already been " ..
		"added to this event logger.")

	local def_name = F'{name}_def'
	local event = lt.message_definition{def_name = def_name, name = name}
	for _, f in pairs(fields) do event:add_field(f) end

	self.event_to_def[name] = def_name
	self.record_def:add_message(event)
end

function event_logger:initialize()
	if paths.filep(self.t7_def_file_path) then
		local old_def = torch.load(self.t7_def_file_path)
		lt.fail_if(self.message_def ~= old_def,
			F"Old event record definition file '{self.t7_def_file_path}' differs " ..
			"from the current definition. Continuing would result in event data in " ..
			"two different formats for the same experiment.", self.logger)
	else
		torch.save(self.t7_def_file_path, self.message_def)
	end

	local proto = self.message_def:to_proto()

	if not paths.filep(self.pb_def_file_path) then
		local f = io.open(self.pb_def_file_path, 'w')
		f:write(proto)
		f:close()
	end

	local def           = pb.load_proto(proto)
	self.factory        = def.Wrapper.EventRecord
	self.wrapper        = def.Wrapper()
	self.cur_record     = def.Wrapper.EventRecord()
	self.wrapper.record = self.cur_record

	self.log_file = io.open(self.log_file_path, 'w')
	self.is_initialized = true
end

function event_logger:register_checkpointer(c)
	-- TODO
end

--[[
Overriden so that users can access the fields of the current event record corresponding to the
events that have been added to this `event_logger`.
--]]
function event_logger:__index__(k)
	local event_to_def = rawget(self, 'event_to_def')

	if event_to_def ~= nil and event_to_def[k] ~= nil then
		local cur_record = rawget(self, 'cur_record')
		local v = cur_record[k]

		if v == nil then
			local factory = rawget(self, 'factory')
			v = factory[event_to_def[k]]()
			cur_record[k] = v
		end

		return v, true
	end

	return false
end

--[[
Overriden so that users can easily set the epoch and iteration values of the current event record.
--]]
function event_logger:__newindex__(k, v)
	if self.required_fields ~= nil and self.required_fields[k] ~= nil then
		self.cur_record[k] = v
		return true
	end

	return false
end

function event_logger:flush()
	for f, _ in pairs(self.required_fields) do
		assert(self.cur_record[f] ~= nil, F"Value for required field '{f}' has not " ..
			"been logged.")
	end

	self.log_file:write(self.wrapper:Serialize())
	self.log_file:flush()
	self.cur_record:Clear()
end
