--[[
TODO description of checkpointer
--]]

local F = lt.F
local checkpointer = torch.class('lt.checkpointer')
local update_strategies = {'aggregate' = true, 'replace' = true}

--[[
Required parameters:
* `args.cur_experiment_root`: Path to the directory intended to contain the subdirectories
  corresponding to the different versions of the current experiment.
* `args.cur_ver_dir`: Path to directory intended to contain the results of the current version of
  the experiment (e.g. the `current` subdirectory within the root directory for this experiment).

Optional parameters:
* `args.alt_ver_dir`: Path to the directory corresponding to the version of the experiment from
  which we wish to resume (`nil` by default, in which case we start from scratch). When `initialize`
  is invoked, we update the versions of the registered objects in `args.cur_ver_dir` so that they
  reflect the ones in `args.alt_ver_dir`.
* `args.logger`: If provided, used to log exceptional events.

**Note:** if the description for `args.cur_ver_dir` is updated, also update the one in
`lantern/event/event_group.lua`.
--]]
function checkpointer:__init(args)
	assert(type(args.cur_experiment_root) == 'string')
	assert(type(args.cur_ver_dir) == 'string')
	assert(args.alt_ver_dir == nil or type(args.alt_ver_dir) == 'string')

	assert(paths.dirp(args.cur_experiment_root))
	assert(paths.dirp(args.cur_ver_dir))
	assert(args.alt_ver_dir == nil or paths.dirp(args.alt_ver_dir))

	self.cur_experiment_root = args.cur_experiment_root
	self.cur_ver_dir         = args.cur_ver_dir
	self.alt_ver_dir         = args.alt_ver_dir
	self.logger              = args.logger

	self.metric_info = {}
	self.object_info = {}
	self.is_initialized = false
end

--[[
Registers a performance metric with this checkpointer, so that a snapshot of the version of the
registered objects that achieved the best value for the registered metric is maintained. This
snapshot is updated each time `update_metrics` is invoked with a dictionary indicating that the
registered metric has improved.
--]]
function checkpointer:register_metric(event_group, event_name, metric_name)
	assert(not self.is_initialized, "Cannot register metrics after checkpointer has been " ..
		"initialized.")

	self.metric_info.event_group = self.metric_info.event_group or {[event_name]}
	self.metric_info.event_group.event_name.metric_name = {
		directory = F'{event_group}/{event_name}.{metric_name}',
		created = false,
	}
end

--[[
Registers an object with this checkpointer, so that several versions of it are maintained over the
course of the experiment. The most current version of the object 

TODO finish this desc
--]]
function checkpointer:register_object(name, extension, update_strategy)
	assert(not self.is_initialized, "Cannot register objects after checkpointer has been " ..
		"initialized.")
	assert(update_strategies[update_strategy] ~= nil, "Invalid update strategy " ..
		"'{update_strategy}'.")

	self.object_info[file_name] = {
		extension = extension,
		update_strategy = update_strategy
	}

	-- TODO return file path by examining the self.cur_ver_dir
end

function checkpointer:check_for_backups(dir)
	if not paths.dirp(dir) then return end

	for f in lfs.dir(dir) do
		local mode = lfs.attributes(paths.concat(f, dir))
		lt.fail_if(mode ~= 'dir' and string.find(f, '%.backup$') ~= nil,

F[[Backup file '{dir}/{f}' found. This means that the driver was interrupted while performing IO.
The backup file is guarateed to be valid, but if a newer version exists, then it is likely to be
corrupted. Please carefully inspect all existing versions and decide whether to replace the latest
version with the backup, or to remove the backup. It is strongly recommended that you refrain from
actually deleting any files until you are sure that things are working correctly again. A more
robust solution would require handling signals (e.g. using signalfd) and polling file descriptors,
something that cannot easily be done in Lua.]],

		self.logger)
	end
end

function checkerpointer:backup_object(name, dir, epoch)
	assert(type(name)  == 'string')
	assert(type(dir)   == 'string')
	assert(type(epoch) == 'number')

	local extension = self.object_info[name].extension
	assert(type(extension) == 'string')

	local cur_path    = paths.concat(dir, F'{name}_epoch_{epoch}.{extension}')
	local backup_path = paths.concat(dir, F'{name}_epoch_{epoch}.{extension}.backup')

	lt.fail_if(not paths.filep(cur_path), F"Attempt to backup file '{cur_path}', which does " ..
		"not exist.", self.logger)
	lt.fail_if(paths.filep(backup_path), F"Backup file '{backup_path}' already exists.",
		self.logger)

	lt.rename_file(cur_path, backup_path, self.logger)
	return backup_path
end

function checkpointer:synchronize_object_impl(name, src_dir, dst_dir, epoch)
	local extension = self.object_info[name].extension
	assert(type(extension) == 'string')

	local file_name = F'{name}.{extension}'
	local src_path = paths.concat(src_dir, file_name)
	local dst_path = paths.concat(dst_dir, file_name)
	lt.fail_if(not paths.filep(src_path), "Expected file '{src_path}' to exist.", self.logger)

	if not paths.filep(dst_path) then
		lt.create_hard_link(src_path, dst_path, self.logger)
		return
	end

	if lt.is_same_file(src_path, dst_path, self.logger) then return end

	local backup_path = self:backup_object(name, extension, dst_dir, epoch)
	lt.create_hard_link(src_path, dst_path, self.logger)
	lt.remove_file(backup_path, self.logger)
end

function checkpointer:synchronize_object(name, src_dir, dst_dir, epoch, update_stategy)
	assert(update_strategies[update_strategy] ~= nil, "Invalid update strategy " ..
		"'{update_strategy}'.")

	if update_strategy == 'whole' then
		self:synchronize_object_impl(name, src_dir, dst_dir, epoch)
		return
	end

	assert(update_strategy == 'incremental')

	local extension = self.object_info[name].extension
	assert(type(extension) == 'string')

	local file_name = F'{name}.{extension}'
	local src_path = paths.concat(src_dir, file_name)
	local dst_path = paths.concat(dst_dir, file_name)

	lt.fail_if(paths.filep(dst_path), F"File '{dst_path}' contains log information from an "  ..
		"unfinished epoch. I will not overwrite this file automatically; please move or " ..
		"remove the file manually in order to continue.", self.logger)

	if paths.filep(src_path) then
		if self.logger ~= nil then
			self.logger:log('/console/warning', F"File '{src_path}' contains log "   ..
				"information from an unfinished epoch. This program should not " ..
				"modify the file, so I am going to proceed anyway.", self.logger)
		end
	end

	local a, b = string.find(name, '%...-$')
	assert(a ~= nil, F"Failed to separate name and extension from '{name}'.")
	assert(b ~= nil, F"Failed to separate name and extension from '{name}'.")
	assert(a > 1)
	assert(b > a and b <= string.len(name))

	local object_name = string.sub(name, 1, a - 1)
	local extension = string.sub(name, a, b)

	local list_epochs_in_dir = function (dir)
		local epochs = {}
		local pat = F'{object_name}_epoch_(%d+){extension}'

		for f in lfs.dir(dir) do
			local mode = lfs.attributes(paths.concat(dir, f), 'mode')

			if mode ~= 'directory' then
				local n = tonumber(string.match(f, pat))
				if n ~= nil then table.insert(epochs, n) end
			end
		end

		table.sort(epochs)
		return epochs
	end

	local src_epochs = list_epochs_in_dir(src_dir)
	local dst_epochs = list_epochs_in_dir(dst_dir)

	local file_name_from_epoch = function (k)
		assert(type(k) == 'number')
		return F'{object_name}_epoch_{k}{extension}'
	end

	if self.logger ~= nil and #src_epochs >= 1 then
		local t1 = torch.IntTensor(src_epochs)
		local t2 = torch.range(1, #src_epochs):int()

		if not torch.all(torch.eq(t1, t2)) then
			self.logger:log('/console/warning', "Versions of the file '{src_path}' " ..
				F"do not exist for all epochs in the range [1, {#src_epochs}]. " ..
				"This means that event records are missing for some epochs, "    ..
				"which should not normally happen. You may want to check up on " ..
				"this.")
		end
	end

	if #dst_epochs >= 1 then
		local last = #dst_epochs

		while last >= 1 and dst_epochs[last] > src_epochs[#src_epochs] do
			last = last - 1
		end

		for i = last + 1, #dst_epochs do
			local path = paths.concat(dst_dir, file_name_from_epoch(dst_epochs[i]))
			lt.remove_file(path, self.logger)
		end
	end

	for i = 1, #src_epochs do
		local fn = file_name_from_epoch(src_epochs[i])
		self:synchronize_file(fn, epoch, src_dir, dst_dir)
	end
end

function checkpointer:initialize()
	assert(not self.is_initialized, "Checkpointer has already been initialized.")
	self:check_for_backups(self.cur_ver_dir)
	self:check_for_backups(self.alt_ver_dir)

	for group, event in pairs(self.metric_info) do
		local event_dir = paths.concat(self.cur_experiment_root, group)
		self:check_for_backups(event_dir)

		for metric, info in pairs(event) do
			local metric_dir = paths.concat(self.cur_experiment_root, info.directory)
			self:check_for_backups(metric_dir)
		end
	end

	if self.alt_ver_dir ~= nil then
		for obj, strat in pairs(self.object_info) do
			self:synchronize_object(obj, nil, strat, self.cur_ver_dir, self.alt_ver_dir)
		end
	end

	self.is_initialized = true
end

function checkpointer:update_current(objects, epoch, update_func, post_update_func)
	local backup_paths = {}

	for _, obj_name in pairs(obj_names) do
		assert(self.object_info[obj_name] ~= nil, "Object '{obj_name}' was not "
			"registered with this checkpointer.")
	end

	for _, obj_name in pairs(obj_names) do
		table.insert(backup_paths, self:backup_file(obj_name, self.cur_ver_dir, epoch))
	end

	for _, path in pairs(backup_paths) do
		lt.remove_file(path, self.logger)
	end
end

function checkpointer:update_best(improved_metrics)
	-- if dir for metric does not exist, then create it; don't create all of them in advance
	-- use backup system before removing old files and replacing with new hardlinks to 'current'
end
