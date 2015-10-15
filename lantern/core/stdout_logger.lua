local json = require "lunajson"
local stdout_logger = lantern.make_class("stdout_logger")

function stdout_logger:__init() end

function stdout_logger:update(resource, data)
	assert(lantern.resources[resource])

	if resource == "/progress" then
		xlua.progress(data.processed_instances, data.total_instances)
	elseif resource == "/console/info" then
		assert(type(data) == "string")
		print(os.date("[%c] ") .. data)
	elseif resource == "/console/warning" then
		assert(type(data) == "string")
		print(os.date("[%c] Warning: ") .. data)
	elseif resource == "/console/error" then
		assert(type(data) == "string")
		print(os.date("[%c] Error: ") .. data)
	end
end
