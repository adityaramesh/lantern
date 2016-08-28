--[[
# Overview

Defines a serialization API that gives the user flexibility over which parts of an object are
serialized, and how the serialization is performed.

# Motivation

Torch's serialization mechanism is inflexible. When an object is serialized, its entire state is
written to disk. This is wasteful if the object owns large internal data structures that are
recreated once the program is restarted. The user also has no control over _how_ the state is
serialized. In many cases, it makes sense to save one part of the object's state in Torch format,
and another part in a format that can easily be parsed in other languages, such as Python. If the
object holds any `CudaTensor`s, then deserialization of the object will be impossible on a system
without enough available GPU memory to reallocate these buffers. There is also no mechanism
available that allows the user to transfer `Tensor`s back to CPU memory before they are saved.

Inheriting `lt.serializable` class addresses these issues by giving the user more control over
the serialization process in the following ways:
1. The user has control over which parts of the object are serialized.
2. Classes can accept additional arguments that are used as part of the serialization and
   deserlization processes.

# Design

By default, the internal state of an object deriving from `serializable`, which contained in
`self.state`, is an empty table. When `torch.save` is invoked on the object, **only this empty
table** is written to the destination file. When `torch.load` is used to deserialize an object, an
instance of the object is created with **only** `self.state` restored.

In order to perform more work or accept additional arguments during the serialization or
deserialization process, the user must override the `__save` and `__load` functions. Each of these
functions accepts a table of arguments called `args`.

For both saving and loading, exactly one of `args.state_file_path` (the path to which the object
should be saved) or `args.state_file` (the file to which the object must be written) must be
provided. The user can require that additional keys be present in `args` when `__save` and `__load`
are invoked on a class deriving from `serializable`

Note the Torch serialization functions are not compatible with `__save` and `__load`, since they
call `__write` and `__read` rather than `__save` and `__load`, respectively. If a derived class
overrides these functions, then `lt.save` and `lt.load` must be used instead. The latter
functions both accept a table of arguments that is forwarded to the implementations of `__save` and
`__load` provided by the derived class.
]]--

local F = lt.F
local serializable = torch.class('lt.serializable')

local TYPE_TABLE = 3
local TYPE_TORCH = 4

local function make_cache_if_not_exists(file)
	if torch.getenv(file).writeObjects then return end

	torch.setenv(file, {
		writeObjects       = {},
		writeObjectsRef    = {},
		readObjects        = {},
		objectNameStack    = {},
		upvalueRefToId     = {},
		upvalueIdToClosure = {},
	})
end

function serializable:__init()
	self.state = self.state or {}
end

function serializable:__write(file)
	assert(self.state ~= nil)
	file:writeObject(self.state)
end

function serializable:__read(file)
	self.state = file:readObject()
	assert(self.state ~= nil)
end

function serializable:__save(args)
	assert(args.state_file ~= nil)

	local file = args.state_file
	make_cache_if_not_exists(file)

	local objects    = torch.getenv(file).writeObjects
	local objectsRef = torch.getenv(file).writeObjectsRef
	local index      = objects[torch.pointer(self.state)]

	if not index then
		-- When we call `writeObject(self.state)`, the `file` object will create a new entry
		-- in `objects` for `self.state`. We increment the index in advance to account for
		-- this.
		index = (objects.nWriteObject or 0) + 1
	end

	local version    = torch.CharStorage():string('V ' .. torch.version(self))
	local class_name = torch.CharStorage():string(torch.typename(self))

	file:writeInt(TYPE_TORCH)
	file:writeInt(index)
	file:writeInt(#version)
	file:writeChar(version)
	file:writeInt(#class_name)
	file:writeChar(class_name)
	self:__write(args.state_file)
end

function serializable:__load(args)
	assert(args.state_file ~= nil)
	self:__read(args.state_file)
end

function lt.save(obj, args)
	assert(args.state_file_path == nil or args.state_file == nil)
	local file = args.state_file

	if args.state_file_path ~= nil then
		file = torch.DiskFile(args.state_file_path, 'w')
		args.state_file = file
	elseif args.state_file == nil then
		error("Either 'args.state_file_path' or 'args.state_file' must be provided.")
	end

	file['binary'](file)
	file:referenced(true)
	obj:__save(args)

	if args.state_file_path ~= nil then file:close() end
end

function lt.load(args)
	assert(args.state_file_path == nil or args.state_file == nil)
	local file = args.state_file

	if args.state_file_path ~= nil then
		file = torch.DiskFile(args.state_file_path, 'r')
		file['binary'](file)
		file:referenced(true)
		args.state_file = file
	elseif args.state_file == nil then
		error("Either 'args.state_file_path' or 'args.state_file' must be provided.")
	end

	local type_index, index = file:readInt(), file:readInt()
	assert(type_index == TYPE_TORCH)

	local objects = torch.getenv(file).readObjects
	if objects[index] then return objects[index] end

	local version = file:readChar(file:readInt()):string()
	local version_number = tonumber(string.match(version, '^V (.*)$'))
	local class_name = file:readChar(file:readInt()):string()

	local factory = torch.factory(class_name)
	assert(factory ~= nil)

	local instance = factory()
	args.state_file = file
	instance:__load(args)

	if args.state_file_path ~= nil then file:close() end
	return instance
end
