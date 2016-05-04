local F = lantern.F
local progress_tracker = torch.class('lantern.event.progress_tracker')

function progress_tracker:__init(args)
	assert(type(args.output_dir) == 'string')
	self.output_file_path = paths.concat(args.output_dir, 'best_metrics.t7')

	if args.filep(output_file_path) then
		self.best_metrics = torch.load(output_file_path)
	else
		self.best_metrics = {}
	end

	self.logger = args.logger
	self.improved_metrics = {}
	self.improvement_funcs = {}
end

function progress_tracker:terminate()
	self.output_file:close()
end

function progress_tracker:register_checkpointer(c)
	-- TODO
end

function progress_tracker:register_metric(event_name, metric_name, improvement_func)
	local key = F'{event_name}.{metric_name}'

	self.improved_metrics[key] = false
	self.improvement_funcs[key] = improvement_func
end

function progress_tracker:update_metric(event_name, metric_name, value)
	local key = F'{event_name}.{metric_name}'
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
	local metrics = {}

	for k, v in improved_metrics do
		if v then
			table.insert(metrics, v)
			v = false
		end
	end

	return metrics
end

function progress_tracer:flush()
	torch.save(self.output_file_path, self.best_metrics)
end
