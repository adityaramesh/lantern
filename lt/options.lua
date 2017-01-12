local F = lt.F

local function default_options(cmd)
	cmd:text("Available options:")
	cmd:option('-task',            'create',      "create | resume | replace")
	cmd:option('-experiment',      '',            "Name of experiment.")
	cmd:option('-gpu',             '1',           "One-indexed GPU ordinal.")
	cmd:option('-experiment_root', 'experiments', "Root directory for experiments.")
end

local tasks = {create = true, resume = true, replace = true}

local function validate_options(options, logger)
	lt.fail_if(tasks[options.task] == nil, "Option 'task' must be one of {'create', " ..
		"'resume', 'replace'}.")

	lt.fail_if(not paths.dirp(options.experiment_root),
		F"Experiment root '{options.experiment_root}' does not exist.")

	lt.fail_if(string.len(options.experiment) == 0, "Experiment name must be provided.")

	lt.fail_if(not lt.is_valid_name(options.experiment),
		lt.invalid_name_msg("Option 'experiment_name'"))

	options.gpu = tonumber(options.gpu)
	lt.fail_if(options.gpu < 0, "Invalid value '{option.gpu}' for parameter 'gpu'.")

	options.experiment_dir = paths.concat(options.experiment_root, options.experiment)

	if options.task == 'create' then
		lt.fail_if(
			paths.dirp(options.experiment_dir),
			F"Experiment directory '{options.experiment_dir}' already exists.",
			logger
		)

		lt.make_directory(options.experiment_dir)
	elseif options.task == 'resume' then
		lt.fail_if(
			not paths.dirp(options.experiment_dir),
			F"Experiment directory '{options.experiment_dir}' does not exist.",
			logger
		)
	elseif options.task == 'replace' then
		lt.fail_if(
			not paths.dirp(options.experiment_dir),
			F"Experiment directory '{options.experiment_dir}' does not exist.",
			logger
		)

		lt.remove_directory(options.experiment_dir)
		lt.make_directory(options.experiment_dir)
	end
end

function lt.parse_options(extra_options)
	local cmd = torch.CmdLine()
	default_options(cmd)
	if extra_options then extra_options(cmd) end

	local options = cmd:parse(arg)
	validate_options(options)
	return options
end
