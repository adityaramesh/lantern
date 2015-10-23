require "lfs"
require "torch"

local json = require "lunajson"
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

function serializer:mkdir_if_not_exists(dir)
	if paths.dirp(dir) then return end

	if not paths.mkdir(dir) then
		self.logger:update(
			"/console/error",
			"Failed creating directory `" .. dir .. "`."
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

	self.logger:update("/console/info", "Restoring backup `" .. old .. "`.")
	self:rename_file_if_exists(old, new)
	return true
end

function serializer:restore_backups()
	local status = true
	status = status and self:restore_backup_if_exists(
		self.cur_model_state_backup_fp, self.cur_model_state_fp)
	status = status and self:restore_backup_if_exists(
		self.cur_opt_state_backup_fp, self.cur_opt_state_fp)
	status = status and self:restore_backup_if_exists(
		self.cur_perf_history_backup_fp, self.cur_perf_history_fp)
	
	for k, v in pairs(self.perf_metrics) do
		for _, mode in pairs(v) do
			status = status and self:restore_backup_if_exists(
				mode.best_model_state_backup_fp, mode.best_model_state_fp)
			status = status and self:restore_backup_if_exists(
				mode.best_opt_state_backup_fp, mode.best_opt_state_fp)
			status = status and self:restore_backup_if_exists(
				mode.best_perf_history_backup_fp, mode.best_perf_history_fp)
		end
	end
	return status
end

function serializer:__init(model_dir, perf_metrics, logger)
	assert(type(model_dir) == "string")
	self:mkdir_if_not_exists(model_dir)
	self.logger = logger or lantern.stdout_logger()

	self.model_dir                  = model_dir
	self.cur_model_state_fp               = paths.concat(model_dir, "model_state_current.t7")
	self.cur_opt_state_fp           = paths.concat(model_dir, "opt_state_current.t7")
	self.cur_perf_history_fp        = paths.concat(model_dir, "perf_history_current.json")
	self.cur_model_state_backup_fp        = paths.concat(model_dir, "model_state_current_backup.t7")
	self.cur_opt_state_backup_fp    = paths.concat(model_dir, "opt_state_current_backup.t7")
	self.cur_perf_history_backup_fp = paths.concat(model_dir, "perf_history_current_backup.json")

	local define_file_paths = function(metric, table)
		table.train = {}
		table.test = {}
	
		table.train.best_model_state_fp = paths.concat(
			model_dir, "model_state_best_train_" .. metric .. ".t7")
		table.train.best_opt_state_fp = paths.concat(
			model_dir, "opt_state_best_train_" .. metric .. ".t7")
		table.train.best_perf_history_fp = paths.concat(
			model_dir, "perf_history_best_train_" .. metric .. ".json")

		table.train.best_model_state_backup_fp = paths.concat(
			model_dir, "model_state_best_train_" .. metric .. "_backup.t7")
		table.train.best_opt_state_backup_fp = paths.concat(
			model_dir, "opt_state_best_train_" .. metric .. "_backup.t7")
		table.train.best_perf_history_backup_fp = paths.concat(
			model_dir, "perf_history_best_train_" .. metric .. "_backup.json")

		table.test.best_model_state_fp = paths.concat(
			model_dir, "model_state_best_test_" .. metric .. ".t7")
		table.test.best_opt_state_fp = paths.concat(
			model_dir, "opt_state_best_test_" .. metric .. ".t7")
		table.test.best_perf_history_fp = paths.concat(
			model_dir, "perf_history_best_test_" .. metric .. ".json")

		table.test.best_model_state_backup_fp = paths.concat(
			model_dir, "model_state_best_test_" .. metric .. "_backup.t7")
		table.test.best_opt_state_backup_fp = paths.concat(
			model_dir, "opt_state_best_test_" .. metric .. "_backup.t7")
		table.test.best_perf_history_backup_fp = paths.concat(
			model_dir, "perf_history_best_test_" .. metric .. "_backup.json")
	end

	self.perf_metrics = {}
	for k, v in pairs(perf_metrics) do
		if v ~= "not important" then
			self.perf_metrics[k] = {}
			define_file_paths(k, self.perf_metrics[k])
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

function serializer:get_improved_metrics(mode, hist)
	assert(hist[#hist][mode])
	if #hist == 1 then return hist[1][mode] end

	local best_metrics = lantern.best_metrics(hist, mode, #hist - 1)
	if #best_metrics == 0 then return end

	local improved = lantern.improved_metrics(best_metrics, hist[#hist][mode])
	if #improved == 0 then return end
	return improved
end 

function serializer:save_current_data(model, optim, hist)
	self:rename_file_if_exists(self.cur_model_state_fp,
		self.cur_model_state_backup_fp)
	self:rename_file_if_exists(self.cur_opt_state_fp,
		self.cur_opt_state_backup_fp)
	self:rename_file_if_exists(self.cur_perf_history_fp,
		self.cur_perf_history_backup_fp)

	torch.save(self.cur_model_state_fp, model.state)
	torch.save(self.cur_opt_state_fp, optim.state)

	local str = json.encode(hist)
	local f = io.open(self.cur_perf_history_fp, 'wb')
	f:write(str):close()

	self:remove_file_if_exists(self.cur_model_state_backup_fp)
	self:remove_file_if_exists(self.cur_opt_state_backup_fp)
	self:remove_file_if_exists(self.cur_perf_history_backup_fp)
end

function serializer:save_current_perf_history(hist)
	self:rename_file_if_exists(self.cur_perf_history_fp,
		self.cur_perf_history_backup_fp)

	local str = json.encode(hist)
	local f = io.open(self.cur_perf_history_fp, 'wb')
	f:write(str):close()

	self:remove_file_if_exists(self.cur_perf_history_backup_fp)
end

function serializer:update_hard_links(mode, metric)
	local paths = self.perf_metrics[metric][mode]

	self:rename_file_if_exists(paths.best_model_state_fp,
		paths.best_model_state_backup_fp)
	self:rename_file_if_exists(paths.best_opt_state_fp,
		paths.best_opt_state_backup_fp)
	self:rename_file_if_exists(paths.best_perf_history_fp,
		paths.best_perf_history_backup_fp)

	self:create_hard_link(self.cur_model_state_fp, paths.best_model_state_fp)
	self:create_hard_link(self.cur_opt_state_fp, paths.best_opt_state_fp)
	self:create_hard_link(self.cur_perf_history_fp, paths.best_perf_history_fp)

	self:remove_file_if_exists(paths.best_model_state_backup_fp)
	self:remove_file_if_exists(paths.best_opt_state_backup_fp)
	self:remove_file_if_exists(paths.best_perf_history_backup_fp)
end

function serializer:save_train_progress(model, optim, hist, logger)
	local improved = self:get_improved_metrics("train", hist)
	self.logger:update("/console/info", "Saving current state.")
	self:save_current_data(model, optim, hist)

	if improved then
		self.logger:update(
			"/console/info",
			"Improved metrics during training: " .. json.encode(improved) .. "."
		)

		for k, v in pairs(improved) do
			self:update_hard_links("train", k)
		end
	end
end

function serializer:save_test_progress(model, optim, hist, logger)
	local improved = self:get_improved_metrics("test", hist)
	self.logger:update("/console/info", "Saving performance history.")
	self:save_current_perf_history(hist)

	if improved then
		self.logger:update(
			"/console/info",
			"Improved metrics during testing: " .. json.encode(improved) .. "."
		)

		for k, v in pairs(improved) do
			self:update_hard_links("test", k)
		end
	end
end
