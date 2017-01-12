local F  = lt.F
local io = lt.io

local tty_logger = torch.class('lt.tty_logger')

function tty_logger:log(resource, data)
	local make_prefix = function(msg)
		local caller_info = debug.getinfo(3)
		local file = lt.path.relpath(caller_info.short_src)
		local line = caller_info.currentline
		local func = caller_info.name

		if msg ~= nil then
			return lt.os.date(F'[%c :: {file}:{line} :: {func}] {msg} ')
		else
			return lt.os.date(F'[%c :: {file}:{line} :: {func}] ')
		end
	end

	--[[
	This function serves two purposes:
	1. Removes line breaks from strings that were supplied using `[[`-style delimiters.
	2. Adds extra newlines in case we are printing to a TTY, so that text does not get swallowed
	   up by the progress bar.
	--]]
	local write = function(msg)
		msg = msg:gsub('[\t\n]', '')
		if lt.term.isatty(io.stdout) then io.write(F'\n{msg}\n\n')
		else io.write(F'{msg}\n') end
	end

	if resource == '/progress' then
		-- We avoid writing out the progress bar animation if stdout is redirected, since
		-- this would unnecessarily increase the file size and make it more difficult to
		-- inspect.
		if lt.term.isatty(io.stdout) then
			xlua.progress(data.current, data.total)
		end
	elseif resource == '/console/info' then
		assert(type(data) == 'string')
		write(make_prefix() .. data)
	elseif resource == '/console/warning' then
		assert(type(data) == 'string')
		write(make_prefix('Warning:') .. data)
	elseif resource == '/console/error' then
		assert(type(data) == 'string')
		write(make_prefix('Error:') .. data)
	else
		error(F"Invalid resource '{resource}'.")
	end
end
