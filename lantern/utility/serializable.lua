--[[
# Overview
Generalizes `flow.serializable` so that the functions that perform the serialization and
deserialization can accept additional arguments.

# Motivation
Some classes need to store parts of their internal state in files separate from the one used by
`torch.save`. Moreover, the names of the files to which the state of the object is written can
change over the course of the program (e.g. when using a `checkpointer`, which incorporates the
current epoch number into the file names). The `__save` and `__load` API generalizes the `__write`
and `__read` API by allowing the user to provide arguments such as file names to the serializable
class.

# Design
By default, `__save` and `__load` expect `args.state_file` to refer to an open file handle (e.g. the
one opened by `torch.save`). The `__save` and `__load` functions pass this handle to the
implementations of `__write` and `__read` provided by `flow.serializable`, respectively.

In order to accept additional arguments, the user must override `__save` and `__load`, optionally
calling the parent class's implementation to serialize `self.state`.
--]]

local serializable, parent = torch.class('lt.serializable', 'flow.serializable')

function serializable:__save(args)
	assert(args.state_file ~= nil)
	parent.__write(self, args.state_file)
end

function serializable:__load(args)
	assert(args.state_file ~= nil)
	parent.__read(self, args.state_file)
end
