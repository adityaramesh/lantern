local F = lt.F
local pb = lt.pb

local event_logger = torch.class('lt.event_logger')
local required_fields = {'epoch' = true, 'iteration' = true}

function event_logger:__init(args)
	assert(type(args.output_dir) == 'string')

	self.state = {
		logger           = args.logger,
		pb_def_file_path = paths.concat(args.output_dir, 'event_record.pb2'),
		message_def      = lt.message_definition{def_name = 'Wrapper'},
		event_to_def     = {}
	}

	self.record_def = lt.message_definition{
		def_name = 'EventRecord',
		name     = 'record',
		rule     = 'required'
	}

	self.state.message_def:add_message(self.record_def)

	for field_name, _ in pairs(required_fields) do
		self.record_def:add_field(lt.field_definition{
			name     = field_name,
			type     = 'uint32',
			required = true
		})
	end

	self.is_initialized = false
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

function event_logger:initialize(args)
	assert(args.checkpointer ~= nil)

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
	self.is_initialized = true

	self.log_file_path = args.checkpointer:register_object('event_data', 'dat', 'incremental')
end

--[[
Closes the current log file, and begins appending records to the new file given by `new_path`
instead. This function should be called by the driver after each epoch.
--]]
function event_logger:update(new_path)
	self.log_file_path = new_path
	self.log_file:close()
	self.log_file = io.open(self.log_file_path, 'w')
end

function event_logger:flush()
	for f, _ in pairs(required_fields) do
		assert(self.cur_record[f] ~= nil, F"Value for required field '{f}' has not " ..
			"been logged.")
	end

	self.log_file:write(self.wrapper:Serialize())
	self.log_file:flush()
	self.cur_record:Clear()
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
	if required_fields[k] ~= nil then
		self.cur_record[k] = v
		return true
	end

	return false
end
