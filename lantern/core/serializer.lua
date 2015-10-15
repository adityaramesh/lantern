require "os"
require "lfs"
require "torch"
local serializer = lantern.make_class("serializer")

function serializer:remove_file_if_exists(fn)
	if not paths.filep(fn) then return end

	local success, err = os.remove(fn)
	if not success then
		self.logger:update(
			"/console/error",
			"Failed removing `" .. old .. "`: " .. err
		)
		os.exit(1)
	end
end

function serializer:rename_file_if_exists(old, new)
	if not paths.filep(old) then return end

	local success, err = os.rename(old, new)
	if not success then
		self.logger:update(
			"/console/error",
			"Failed renaming `" .. old .. "` to `" .. new .. "`: " .. err
		)
		os.exit(1)
	end
end

function serializer:create_hard_link(src, dst)
	local success, err = lfs.link(src, dst)
	if not success then
		self.logger:update(
			"/console/error",
			"Failed creating hard link `" .. dst .. "`: " .. err
		)
		os.exit(1)
	end
end

function serializer:restore_backup_if_exists(old, new)
	if not paths.filep(old) then
		return true
	elseif paths.filep(old) and paths.filep(new) then
		self.logger:update(
			"/console/error",
			"Both `" .. old ..  "` and `" ..  new .. "` exist. The "     ..
			"driver was likely interrupted while writing to the latter " ..
			"file. Please carefully inspect both files, and either "     ..
			"replace the latter file with the former, or delete the "    ..
			"former."
		)
		return false
	end

	self:rename_file_if_exists(old, new)
	return true
end

function serializer:restore_backups()
	local status = true
	status = status and self:restore_backups(
		self.cur_model_backup_fp, self.cur_model_fp)
	status = status and self:restore_backups(
		self.cur_opt_state_backup_fp, self.cur_opt_state_fp)
	status = status and self:restore_backups(
		self.cur_driver_state_backup_fp, self.cur_driver_state_fp)
	
	for k, v in pairs(self.perf_metrics) do
		for _, mode in pairs(v) do
			status = status and self:restore_backups(
				self.best_model_backup_fp, self.best_model_fp)
			status = status and self:restore_backups(
				self.best_opt_state_backup_fp, self.best_opt_state_fp)
			status = status and self:restore_backups(
				self.best_driver_state_backup_fp, self.best_driver_state_fp)
		end
	end
	return status
end

function serializer:__init(model_dir, perf_metrics)
	assert(type(model_dir) == "string")
	assert(
		paths.dirp(model_dir),
		"The directory `" .. model_dir .. "` does not exist."
	)

	self.model_dir                  = model_dir
	self.cur_model_fp               = paths.concat(model_dir, "model_current.t7")
	self.cur_opt_state_fp           = paths.concat(model_dir, "opt_state_current.t7")
	self.cur_driver_state_fp        = paths.concat(model_dir, "driver_state_current.json")
	self.cur_model_backup_fp        = paths.concat(model_dir, "model_current_backup.t7")
	self.cur_opt_state_backup_fp    = paths.concat(model_dir, "opt_state_current_backup.t7")
	self.cur_driver_state_backup_fp = paths.concat(model_dir, "driver_state_current_backup.json")

	local define_file_paths = function(metric, table)
		table.train = {}
		table.test = {}
	
		table.train.best_model_fp = paths.concat(
			model_dir, "model_best_train_" .. metric .. ".t7")
		table.train.best_opt_state_fp = paths.concat(
			model_dir, "opt_state_best_train_" .. metric .. ".t7")
		table.train.best_driver_state_fp = paths.concat(
			model_dir, "driver_state_best_train_" .. metric .. ".json")

		table.train.best_model_backup_fp = paths.concat(
			model_dir, "model_best_train_" .. metric .. "_backup.t7")
		table.train.best_opt_state_backup_fp = paths.concat(
			model_dir, "opt_state_best_train_" .. metric .. "_backup.t7")
		table.train.best_driver_state_backup_fp = paths.concat(
			model_dir, "driver_state_best_train_" .. metric .. "_backup.json")

		table.test.best_model_fp = paths.concat(
			model_dir, "model_best_test_" .. metric .. ".t7")
		table.test.best_opt_state_fp = paths.concat(
			model_dir, "opt_state_best_test_" .. metric .. ".t7")
		table.test.best_driver_state_fp = paths.concat(
			model_dir, "driver_state_best_test_" .. metric .. ".json")

		table.test.best_model_backup_fp = paths.concat(
			model_dir, "model_best_test_" .. metric .. "_backup.t7")
		table.test.best_opt_state_backup_fp = paths.concat(
			model_dir, "opt_state_best_test_" .. metric .. "_backup.t7")
		table.test.best_driver_state_backup_fp = paths.concat(
			model_dir, "driver_state_best_test_" .. metric .. "_backup.json")
	end

	self.perf_metrics = {}
	for k, v in pairs(perf_metrics) do
		if v ~= "not important" then
			self.perf_metrics[k] = {}
			define_file_paths(v, self.perf_metrics[v])
		end
	end

	if not self:restore_backups() then
		self.logger:update(
			"/console/error",
			"Both backup and non-backup versions of files exist. " ..
			"Will not proceed until ambiguities are resolved."
		)
		os.exit(1)
	end
end

function serializer:load_state()
	if not paths.filep(self.cur_driver_state_fp) then return end

	local f = io.open(self.cur_driver_state_fp, 'rb')
	local text = f:read("*all"):close()
	return json.decode(text)
end

function serializer:get_improved_metrics(mode, state)
	local prev_metrics = nil
	local cur_metrics = state[#state][mode]
	assert(cur_metrics)

	for i = #state - 1, 1, -1 do
		if state[i][mode] then
			prev_metrics = state[i][mode]
			break
		end
	end

	local improved_metrics = {}
	for k, _ in pairs(self.perf_metrics) do
		improved_metrics[#improved_metrics + 1] = k
	end

	if not prev_metrics then
		return improved_metrics
	end

	for k, v in pairs(cur_metrics) do
		if lantern.performance_metrics[k] == "increasing" then
			if v <= prev_metrics[k] then
				improved_metrics[k] = nil
			end
		elseif v >= prev_metrics[k] then
			improved_metrics[k] = nil
		end
	end

	return improved_metrics
end

function serializer:save_current_data(model, optim, state)
	self:rename_file_if_exists(self.cur_model_fp,
		self.cur_model_backup_fp)
	self:rename_file_if_exists(self.cur_opt_state_fp,
		self.cur_opt_state_backup_fp)
	self:rename_file_if_exists(self.cur_driver_state_fp,
		self.cur_driver_state_backup_fp)

	torch.save(self.cur_model_fp, model)
	torch.save(self.cur_opt_state_fp, optim)

	local str = json.encode(state)
	local f = io.open(self.cur_driver_state_fp, 'wb')
	f:write(str):close()

	self:remove_file_if_exists(self.cur_model_backup_fp)
	self:remove_file_if_exists(self.cur_opt_state_backup_fp)
	self:remove_file_if_exists(self.cur_driver_state_backup_fp)
end

function serializer:update_hard_links(mode, metric)
	local paths = self.perf_metrics[metrics][mode]

	self:rename_file_if_exists(paths.best_model_fp,
		paths.best_model_backup_fp)
	self:rename_file_if_exists(paths.best_opt_state_fp,
		paths.best_opt_state_backup_fp)
	self:rename_file_if_exists(paths.best_driver_state_fp,
		paths.best_driver_state_backup_fp)

	self:create_hard_link(self.cur_model_fp, paths.best_model_fp)
	self:create_hard_link(self.cur_opt_state_fp, paths.best_opt_state_fp)
	self:create_hard_link(self.cur_driver_state_fp, paths.best_driver_state_fp)

	self:remove_file_if_exists(paths.best_model_backup_fp)
	self:remove_file_if_exists(paths.best_opt_state_backup_fp)
	self:remove_file_if_exists(paths.best_driver_state_backup_fp)
end

function serializer:save_train_progress(model, optim, state, logger)
	local improved = self:get_improved_metrics("train", state)
	self:save_current_data(model, optim, state)

	for _, v in pairs(improved) do
		self:update_hard_links("train", v)
	end
end

function serializer:save_test_progress(model, optim, state, logger)
	local improved = self:get_improved_metrics("test", state)

	for _, v in pairs(improved) do
		self:update_hard_links("test", v)
	end
end
