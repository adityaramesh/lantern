--[[
Defines the essential command-line options for lantern.

Since training and evaluating a model typically involve different data sets and configurations for
the driver, it makes more sense to use separate scripts for the two tasks rather than to hack one
script into doing both. For this reason, the command-line option 'task' only deals with creating,
resuming, and starting, experiments, rather than determining what they should do.
--]]

function lantern.default_options(cmd)
	cmd:text("Available options:")
	cmd:option("-task",            'create',      "create | resume | replace")
	cmd:option("-experiment",      '',            "Name of experiment.")
	cmd:option("-version",         'current',     "Version of experiment to resume.")
	cmd:option("-gpus",            '1',           "Comma-separated list of GPU ordinals.")
	cmd:option("-experiment_root", 'experiments', "Root directory for experiments.")
end

local function validate_options(options, logger)
	lantern.fail_if({'create', 'resume', 'replace'}[options.task] == nil,
		"Option 'task' must be one of {'create', 'resume', 'replace'}.")

	lantern.fail_if(not paths.dirp(options.experiment_root),
		F"Experiment root '{options.experiment_root}' does not exist.")

	lantern.fail_if(string.len(options.experiment) == 0, "Experiment name must be provided.")

	lantern.fail_if(not lantern.is_valid_name(options.experiment),
		lantern.invalid_name_msg("Option 'experiment_name'"))

	local gpu_list = {}

	for s in string.gmatch(options.gpus, '([^%s,]+)') do
		local i = tonumber(s)
		lantern.fail_if(i <= 0, "GPU ordinals must be positive integers.")
		gpu_list[#gpu_list + 1] = i
	end

	options.gpus = gpu_list

	local exp_dir = paths.concat(options.experiment_root, options.experiment)
	local exp_ver_dir = paths.concat(options.experiment_root, options.experiment,
		options.version)

	if options.task == 'create' then
		lantern.fail_if(
			paths.dirp(exp_dir),
			F"Experiment directory '{exp_dir}' already exists.",
			logger
		)

		lantern.make_directory(exp_dir)
	elseif options.task == 'resume' then
		lantern.fail_if(
			not paths.dirp(exp_dir),
			F"Experiment directory '{exp_dir}' does not exist.",
			logger
		)
		lantern.fail_if(
			string.len(options.version) == 0,
			"Experiment version must be provided.",
			logger
		)
		lantern.fail_if(
			not paths.dirp(exp_ver_dir),
			F"Experiment version '{exp_ver_dir}' does not exist.",
			logger
		)
	elseif options.task == 'replace' then
		lantern.fail_if(
			not paths.dirp(exp_dir),
			F"Experiment directory '{exp_dir}' does not exist.",
			logger
		)

		lantern.remove_directory(exp_dir)
		lantern.make_directory(exp_dir)
	end
end

function lantern.parse_options(extra_options)
	local cmd = torch.CmdLine()
	lantern.default_options(cmd)
	if extra_options then extra_options(cmd) end

	local options = cmd:parse(arg)
	validate_options(options)
	return options
end
