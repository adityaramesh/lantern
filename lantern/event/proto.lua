--[[
Utilities for automatically generating protobuf definitions.
--]]

local F = lt.F
local field_definition = torch.class('lt.field_definition')
local message_definition = torch.class('lt.message_definition')

local scalar_types = {double = true, float = true, int32 = true, int64 = true, uint32 = true,
	uint64 = true, sint32 = true, sint64 = true, fixed32 = true, fixed64 = true,
	sfixed32 = true, sfixed64 = true, bool = true}

local function is_scalar_type(t)
	return scalar_types[t] ~= nil
end

--[[
Required parameters:
* `args.name`: The name of this field.
* `args.type`: The type of this field.

Optional parameters:
* `args.rule`: The rule associated with the field (either 'required', 'optional' or 'repeated').
  Defaults to 'optional'.
--]]
function field_definition:__init(args)
	assert(type(args.name) == 'string')
	assert(type(args.type) == 'string')

	self.name = args.name
	self.type = args.type
	self.rule = args.rule or 'optional'
end

function field_definition:__eq(rhs)
	return self.name == rhs.name and self.type == rhs.type and
		self.rule == rhs.rule
end

--[[
Required parameters:
* `args.name`: The name of this message.
* `args.def_name`: The name of this message definition.

Optional parameters:
* `args.rule`: The rule associated with the message (either 'required', 'optional' or 'repeated').
  Defaults to 'optional'.
--]]
function message_definition:__init(args)
	assert(type(args.def_name) == 'string')

	self.name     = args.name
	self.def_name = args.def_name
	self.rule     = args.rule or 'optional'

	self.fields        = {}
	self.messages      = {}
	self.name_to_field = {}
	self.name_to_msg   = {}
end

function message_definition:add_field(field)
	assert(self.name_to_field[field.name] == nil, F"Field with name {field.name} already exists.")

	self.name_to_field[field.name] = field
	table.insert(self.fields, field)
end

function message_definition:add_message(msg)
	assert(msg.name ~= nil, "Nested message must specify name for instance.")
	assert(self.name_to_msg[msg.def_name] == nil, F"Message with name {msg.name} already " ..
		"exists.")

	self.name_to_msg[msg.def_name] = msg
	table.insert(self.messages, msg)
end

function message_definition:__eq(rhs)
	if not (#self.fields   == #rhs.fields   and
	        #self.messages == #rhs.messages and
	        self.name      == rhs.name      and
		self.def_name  == rhs.def_name) then return false end

	for k, v in pairs(self.name_to_field) do
		if rhs.name_to_field[k] == nil or v ~= rhs.name_to_field[k] then return false end
	end

	for k, v in pairs(self.name_to_msg) do
		if rhs.name_to_msg[k] == nil or v ~= rhs.name_to_msg[k] then return false end
	end

	return true
end

function message_definition:to_proto(cur_indent, lines)
	local cur_indent = cur_indent or 0
	local indent_1   = nil
	local indent_2   = string.rep('\t', cur_indent + 1)
	
	if cur_indent >= 1 then
		indent_1 = string.rep('\t', cur_indent)
	else
		indent_1 = ''
	end

	local tag = 1
	local lines = lines or {}
	table.insert(lines, F'{indent_1}message {self.def_name} {')

	for _, m in pairs(self.messages) do
		m:to_proto(cur_indent + 1, lines)
	end

	for i, f in pairs(self.fields) do
		local rule = f.rule

		if not (f.rule == 'repeated' and is_scalar_type(f.type)) then
			table.insert(lines, F'{indent_2}{f.rule} {f.type} {f.name} = {i};')
		else
			table.insert(lines, F'{indent_2}repeated {f.type} {f.name} = {i} ' ..
				'[packed=true];')
		end
	end

	for i, m in pairs(self.messages) do
		local rule = m.rule

		if not (m.rule == 'repeated' and is_scalar_type(m.type)) then
			table.insert(lines,
				F'{indent_2}{m.rule} {m.def_name} {m.name} = {i + #self.fields};')
		else
			table.insert(lines, F'{indent_2}repeated {m.def_name} {m.name} = ' ..
				F'{i + #self.fields} [packed=true];')
		end
	end

	table.insert(lines, F'{indent_1}}')
	return table.concat(lines, '\n')
end
