--[[
Utilities for automatically generating protobuf definitions.
--]]

local F = lt.F
local sys_stat = lt.sys_stat
local field_definition = torch.class('lt.field_definition')
local message_definition = torch.class('lt.message_definition')

--[[
Required parameters:
* `args.name`: The name of this field.
* `args.type`: The type of this field.

Optional parameters:
* `args.required`: Whether this field is required (false by default).
--]]
function field_definition:__init(args)
	assert(type(args.name) == 'string')
	assert(type(args.type) == 'string')

	self.name = args.name
	self.type = args.type
	self.is_required = args.required or false
end

function field_definition:__eq(rhs)
	return self.name == rhs.name and self.type == rhs.type and
		self.is_required == rhs.is_required
end

--[[
Required parameters:
* `args.name`: The name of this message.

Optional parameters:
* `args.required`: Whether this field is required (false by default).
--]]
function message_definition:__init(args)
	assert(type(args.name) == 'string')

	self.name = args.name
	self.is_required = args.required or false

	self.fields = {}
	self.messages = {}
	self.name_to_field = {}
	self.name_to_msg = {}
end

function message_definition:add_field(field)
	assert(self.name_to_field[field.name] == nil, F"Field with name {field.name} already exists.")

	self.name_to_field[field.name] = field
	table.insert(self.fields, field)
end

function message_definition:add_message(msg)
	assert(self.name_to_msg[msg.name] == nil, F"Message with name {msg.name} already exists.")

	self.name_to_msg[msg.name] = msg
	table.insert(self.messages, msg)
end

function message_definition:__eq(rhs)
	if #self.fields ~= #rhs.fields or #self.messages ~= #rhs.messages then return false end

	for k, v in self.name_to_field do
		if rhs.name_to_field[k] == nil or v ~= rhs.name_to_field[k] then return false end
	end

	for k, v in self.name_to_msg do
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
	table.insert(lines, F'{indent_1}message {self.name} {')

	for _, m in pairs(self.messages) do
		m:to_proto(cur_indent + 1, lines)
	end

	for i, f in pairs(self.fields) do
		local attr = nil
		if f.is_required then attr = 'required' else attr = 'optional' end
		table.insert(lines, F'{indent_2}{attr} {f.type} {f.name} = {i};')
	end

	for i, m in pairs(self.messages) do
		local attr = nil
		if m.is_required then attr = 'required' else attr = 'optional' end
		table.insert(lines, F'{indent_2}{attr} {m.name} m_{i} = {i + #self.fields};')
	end

	table.insert(lines, F'{indent_1}}')
	return table.concat(lines, '\n')
end
