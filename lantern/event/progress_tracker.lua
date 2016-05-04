local F = lantern.F
local mp = lantern.mp
local progress_tracker = torch.class('lantern.event.progress_tracker')

function progress_tracker:__init(args)
	assert(type(args.output_dir) == 'string')
	local output_file_name = 'best_metrics.dat'
	self.logger = args.logger

	if args.input_dir ~= nil then
		assert(paths.dirp(args.input_dir))
		local input_file_path = paths.concat(args.input_dir, output_file_name)

		if not paths.filep(input_file_path) and self.logger ~= nil then
			self.logger:log('/console/warning', "Directory {args.input_dir} exists, " ..
				"but the file {input_file_path} does not.")
		else
			local f = io.open(input_file_path, 'r')
			self.best_metrics = mp.unpack(f:read())
			f:close()
		end
	end

	self.best_metrics = self.best_metrics or {}
	self.improved_metrics = {}
	self.improvement_funcs = {}

	self.output_file_path = paths.concat(args.output_dir, output_file_name)
end

function progress_tracker:terminate()
	self.output_file:close()
end

function progress_tracker:register_metric(event_name, metric_name, improvement_func)
	local key = F'{event_name}/{metric_name}'

	self.improved_metrics[key] = false
	self.improvement_funcs[key] = improvement_func
end

function progress_tracker:update_metric(event_name, metric_name, value)
	local key = F'{event_name}/{metric_name}'
	local old_value = self.best_metrics[key]

	if old_value == nil then
		self.best_metrics[key] = value
		self.improved_metrics[key] = true
	else
		local improved = self.improvement_func[key](old_value, value)

		if improved then
			self.best_metrics[key] = value
			self.improved_metrics[key] = true
		end
	end
end

function progress_tracker:improved_metrics()
	local f = io.open(output_file_path, 'w')
	f:write(mp.pack(self.best_metrics))
	f:close()

	local metrics = {}

	for k, v in improved_metrics do
		if v then
			table.insert(metrics, v)
			v = false
		end
	end

	return metrics
end
