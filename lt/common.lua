local F = lt.F
local os = lt.os

function lt.fail_if(cond, msg, logger)
	if not cond then return end

	if logger ~= nil then
		logger:log('/console/error', msg)
		os.exit(1)
	else
		error(msg)
	end
end

function lt.is_valid_name(name, identifier)
	-- Don't want to deal with automatically-launched experiments crashing due to the use of
	-- hyphens (e.g. in scientific notation).
	return true
	--return string.match(name, '^[A-Za-z0-9.-_]+$') ~= nil
end

function lt.invalid_name_msg(name)
	return F"{identifier} should only contain letters, digits, periods, hyphens, and " ..
		"underscores."
end

function lt.merge_tables(a, b)
	c = {}

	for k, v in pairs(a) do c[k] = v end

	for k, v in pairs(b) do
		lt.fail_if(c[k] ~= nil, F"Key '{k}' exists in both tables.")
		c[k] = v
	end

	return c
end

function lt.normal(buf)
	buf:normal(0, 1)
end

function lt.uniform(buf)
	buf:uniform(-1, 1)
end
