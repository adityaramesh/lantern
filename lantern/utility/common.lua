local os = flow.os
local F = lantern.F

function lantern.fail_if(cond, msg, logger)
	if logger ~= nil then
		logger:log('/console/error', msg)
		os.exit(1)
	else
		error(msg)
	end
end

function lantern.is_name_valid(name, identifier)
	return string.match(name, '^[A-Za-z0-9.-_]+$') ~= nil
end

function lantern.invalid_name_msg(name)
	return F"{identifier} should only contain letters, digits, periods, hyphens, and " ..
		"underscores."
end
