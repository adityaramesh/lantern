local csv_logger = lantern.make_class("csv_logger")

function csv_logger:__init(file_path, fields)
	self.file = io.open(file_path, "a")

	self.fields  = {}
	self.indices = {}
	self.cur_row = {}
	if fields ~= nil then self:add_fields(fields) end
end

function csv_logger:add_fields(fields)
	for k = 1, #fields do
		self.fields[#self.fields + 1] = fields[k]
		self.indices[fields[k]] = #self.fields
		self.cur_row[#self.cur_row + 1] = "\"\""
	end
end

function csv_logger:write_header()
	self.file:write(table.concat(self.fields, ", "), "\n")
	self.file:flush()
end

function csv_logger:log_value(field, value)
	assert(self.indices[field] ~= nil)

	local index = self.indices[field]
	self.cur_row[index] = tostring(value)
end

function csv_logger:log_array(field, values)
	assert(self.indices[field] ~= nil)

	local index = self.indices[field]
	self.cur_row[index] = "\"" .. table.concat(values, ", ") .. "\""
end

function csv_logger:flush()
	self.file:write(table.concat(self.cur_row, ", "), "\n")
	self.file:flush()

	for k = 1, #self.cur_row do
		self.cur_row[k] = "\"\""
	end
end

function csv_logger:close()
	self.file:close()
end
