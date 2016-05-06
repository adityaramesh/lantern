local F = lt.F
local os = lt.os
local lfs = lt.lfs
local ffi = lt.ffi

function lt.fail_if(cond, msg, logger)
	if not cond then return end

	if logger ~= nil then
		logger:log('/console/error', msg)
		os.exit(1)
	else
		error(msg)
	end
end

function lt.is_name_valid(name, identifier)
	return string.match(name, '^[A-Za-z0-9.-_]+$') ~= nil
end

function lt.invalid_name_msg(name)
	return F"{identifier} should only contain letters, digits, periods, hyphens, and " ..
		"underscores."
end

function lt.load_binary(path)
	local size = lfs.attributes(path).size
	local f = io.open(path, 'rb')
	local buf = ffi.new('uint8_t[?]', size)

	ffi.copy(buf, f:read(size), size)
	f:close()
	return ffi.string(buf, size)
end
