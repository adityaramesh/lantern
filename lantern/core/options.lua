require "lfs"
require "xlua"
require "cutorch"
local json = require "lunajson"

local function make_directory(dir)
	local success, err = lfs.mkdir(dir)

	if not success then
		error("Failed to make directory `" .. dir .. "`: " .. err)
	end
end

local function remove_empty_directory(dir)
	if not paths.dirp(dir) then
		error("Could not find directory `" .. dir .. "`.")
	end

	local success, err = lfs.rmdir(dir)
	if not success then
		error("Failed to remove directory `" .. dir .. "`: " .. err)
	end
end

local function remove_directory(dir)
	for file in lfs.dir(dir) do
		local path = paths.concat(dir, file)
		if lfs.attributes(path, "mode") ~= "directory" then
			local success, err = os.remove(path)

			if not success then
				error("Failed to remove file `" .. file .. "`: " .. err)
			end
		end
	end

	remove_empty_directory(dir)
end

function lantern.default_options(cmd)
	cmd:text("Select one of the following options:")
	cmd:option("-task",        "create", "create | resume | replace")
	cmd:option("-model",       "test",   "Name of model.")
	cmd:option("-device",      1,        "GPU device number.")
	cmd:option("-model_state", "",       "Path to T7 file with model state.")
	cmd:option("-optim_state", "",       "Path to T7 file with optimizer state.")
	cmd:option("-hist_file",   "",       "Path to JSON file with performance history.")
	cmd:option("-models_dir",  "models", "Root directory for models.")
end

--
-- Applies the required command-line arguments using `lantern.default_options`,
-- and sets the variable `lantern.options` to the table of parsed command-line
-- arguments. Returns a table containing the following:
-- * model_dir: The directory in which the model information is saved.
-- * model:     The model state (null if not provided).
-- * optimizer: The optimizer state (null if not provided).
-- * history:   The driver state (null if not provided).
--
function lantern.parse_options(extra_options)
	local cmd = torch.CmdLine()
	lantern.default_options(cmd)
	if extra_options then extra_options(cmd) end

	lantern.options = cmd:parse(arg)
	local opt = lantern.options

	cutorch.setDevice(opt.device)
	torch.manualSeed(1)
	cutorch.manualSeed(1)
	-- Fail fast if something is wrong with the GPU.
	torch.zeros(1, 1):cuda():uniform()

	local models_dir = opt.models_dir
	if not paths.dirp(models_dir) then
		make_directory(models_dir)
	end

	if string.match(opt.model, "^[A-Za-z0-9_]+") == nil then
		error("Invalid model name `" .. opt.model .. "`.")
	end

	local model_path = opt.model_state
	local optim_path = opt.optim_state
	local hist_path  = opt.hist_file
	local model_dir  = paths.concat(models_dir, opt.model)

	if opt.task == "resume" then
		if not paths.dirp(model_dir) then
			error("Model directory `" .. opt.model .. "` does not exist.")
		end

		model_path = paths.concat(model_dir, "model_state_current.t7")
		optim_path = paths.concat(model_dir, "opt_state_current.t7")
		hist_path  = paths.concat(model_dir, "perf_history_current.json")
	elseif opt.task == "create" then
		if paths.dirp(model_dir) then
			error("Model directory `" .. opt.model .. "` already exists.")
		end
		make_directory(model_dir)
	elseif opt.task == "replace" then
		if paths.dirp(model_dir) then
			remove_directory(model_dir)
		end
		make_directory(model_dir)
	else
		error("Invalid task `" .. opt.task .. "`.")
	end

	local load_state = function(path)
		if path == "" then return end
		local ext = paths.extname(path)

		if ext ~= "t7" then
			error("Invalid extension for file `" .. path .. "`.")
		end
		return torch.load(path)
	end

	local load_json = function(path)
		if path == "" then return end
		local ext = paths.extname(path)

		if ext ~= "json" then
			error("Invalid extension for file `" .. path .. "`.")
		end

		local f = io.open(path, 'rb')
		local text = f:read("*all")
		f:close()

		return json.decode(text)
	end

	return {
		model_dir   = model_dir,
		model_state = load_state(model_path),
		optim_state = load_state(optim_path),
		history     = load_json(hist_path)
	}
end
