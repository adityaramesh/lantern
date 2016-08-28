local F = lt.F
local checkpointer = torch.class('lt.checkpointer')

local function make_file_path(output_dir, name, epoch)
	return paths.concat(output_dir, F'{name}.epoch_{epoch}.t7')
end

function checkpointer:__init(args)
	assert(paths.dirp(args.output_dir))
	assert(type(args.objects) == 'table')

	self.output_dir   = args.output_dir
	self.object_names = {}
	self.object_count_ = #args.objects

	for _, obj_name in pairs(args.objects) do
		self.object_names[obj_name] = true
	end

	local obj_file_pat    = '^([A-Za-z0-9-_]+)%.epoch_(%d+)%.t7$'
	local backup_file_pat = '.+%.backup$'
	local name_to_epoch   = {}
	local obj_found       = 0

	for file in lfs.dir(args.output_dir) do
		local obj_name, epoch = string.match(file, obj_file_pat)
		epoch = tonumber(epoch)

		if obj_name ~= nil and epoch ~= nil then
			if epoch < 1 then
				error(F"File '{args.output_dir}/{file}' has invalid epoch " ..
					F"number {epoch}.")
			end

			if self.object_names[obj_name] ~= nil then
				local prev_epoch = name_to_epoch[obj_name]

				if prev_epoch == nil then
					name_to_epoch[obj_name] = epoch
					obj_found = obj_found + 1
				else
					name_to_epoch[obj_name] = math.max(prev_epoch, epoch)
				end
			end
		elseif string.match(file, backup_file_pat) ~= nil then
			error(F"Backup file '{args.output_dir}/{file}' exists. This means that " ..
				"the checkpointer was interrupted while performing IO. Please "  ..
				"move or delete the file manually in order to proceed.")
		end
	end

	if obj_found >= 1 and obj_found ~= #args.objects then
		error("Could not find saved versions of all registered objects.")
	end

	self.last_epoch_ = nil

	for k, v in pairs(name_to_epoch) do
		if self.last_epoch_ ~= nil then
			if self.last_epoch_ ~= v then
				error("The most current versions of the saved objects do not " ..
					"correspond to the same epoch.")
			end
		else
			self.last_epoch_ = v
		end
	end
end

function checkpointer:object_count()
	return self.object_count_
end

function checkpointer:last_epoch()
	return self.last_epoch_
end

function checkpointer:__index__(k)
	local object_names = rawget(self, 'object_names')

	if object_names ~= nil and object_names[k] ~= nil then
		local epoch = self:last_epoch()

		if epoch ~= nil then
			return lt.load{state_file_path = make_file_path(
				self.output_dir, k, self:last_epoch())}, true
		else
			return nil, true
		end
	end

	return false
end

function checkpointer:update(epoch, objects)
	if self:last_epoch() then
		assert(epoch > self:last_epoch())
	else
		assert(epoch == 1)
	end

	local count = 0

	for obj_name, _ in pairs(objects) do
		assert(self.object_names[obj_name] ~= nil)
		count = count + 1
	end

	assert(count == self:object_count())

	local prev_epoch = self:last_epoch()
	self.last_epoch_ = epoch

	for name, obj in pairs(objects) do
		lt.save(obj, {
			epoch = epoch,
			state_file_path = make_file_path(self.output_dir, name, self:last_epoch())
		})
	end

	if prev_epoch ~= nil then
		for name, _ in pairs(objects) do
			lt.remove_file(make_file_path(self.output_dir, name, prev_epoch))
		end
	end
end
