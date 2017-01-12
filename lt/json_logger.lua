local F = lt.F
local json_logger, parent = torch.class('lt.json_logger', 'lt.serializable')

function json_logger:__init(args)
	assert(type(args.fields) == 'table')
	assert(type(args.output_dir) == 'string')
	assert(paths.dirp(args.output_dir))

	parent.__init(self)
	self.state.fields = {}

	for _, f in pairs(args.fields) do
		self.state.fields[f] = true
	end

	self.state.output_dir = args.output_dir
	self.state.output_file_path = paths.concat(args.output_dir, 'log.json')
	self:initialize()
end

function json_logger:initialize()
	self.cur_msg     = {}
	self.msg_buf     = {}
	self.first_entry = true
	self.output_file = io.open(self.state.output_file_path, 'w')
	self.output_file:write('[\n')
end

function json_logger:__index__(k)
	local state   = rawget(self, 'state')
	local cur_msg = rawget(self, 'cur_msg')
	local msg_buf = rawget(self, 'msg_buf')

	if state ~= nil and cur_msg ~= nil and msg_buf ~= nil then
		if type(k) == 'string' then
			if state.fields[k] ~= nil then return cur_msg[k], true end
			return false
		elseif type(k) == 'number' then
			assert(k <= #self.msg_buf + 1)
			if k <= #self.msg_buf then return self.msg_buf[k], true end
			return cur_msg, true
		else
			error(F"Invalid key '{k}' of type '{type(k)}'.")
		end
	end

	return false
end

function json_logger:__newindex__(k, v)
	if self.state ~= nil and self.state.fields ~= nil and self.state.fields[k] ~= nil then
		self.cur_msg[k] = v
		return true
	end

	return false
end

function json_logger:commit()
	table.insert(self.msg_buf, self.cur_msg)
	self.cur_msg = {}
end

function json_logger:flush()
	local cur_msg_len = 0
	for k, v in pairs(self.cur_msg) do cur_msg_len = cur_msg_len + 1 end
	assert(cur_msg_len == 0, "Attempt to flush while there is an uncommitted message.")

	if #self.msg_buf == 0 then return end
	local lines = {}

	for _, msg in pairs(self.msg_buf) do
		if self.first_entry then
			table.insert(lines, '{')
			self.first_entry = false
		else
			table.insert(lines, ', {')
		end

		local index = 1
		local msg_len = 0
		for k, v in pairs(msg) do msg_len = msg_len + 1 end

		for k, v in pairs(msg) do
			if index == msg_len then
				table.insert(lines, F'\t\"{k}\": {v}')
			else
				table.insert(lines, F'\t\"{k}\": {v},')
			end

			index = index + 1
		end

		table.insert(lines, "}")
	end

	table.insert(lines, "")
	self.output_file:write(table.concat(lines, '\n'))

	self.msg_buf = {}
	self.cur_msg = {}
end

function json_logger:__save(args)
	assert(type(args.epoch) == 'number')

	local path = paths.concat(self.state.output_dir, F'log.epoch_{args.epoch}.json')
	assert(not paths.filep(path))

	self.output_file:write(']')
	self.output_file:close()
	lt.rename_file(self.state.output_file_path, path)
	self:initialize()

	parent.__save(self, args)
end

function json_logger:__load(args)
	parent.__load(self, args)
	self:initialize()
end
