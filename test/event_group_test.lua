require('lt')
require('totem')

local F = lt.F
local tests = totem.TestSuite()
local tester = totem.Tester()

function tests.test_logging_and_tracking()
	local test_dir = 'scratch/dummy_experiment'
	local output_dir = paths.concat(test_dir, 'current')

	lt.make_directory_if_not_exists(test_dir)
	lt.make_directory_if_not_exists(output_dir)

	local g = lt.event_group{name = 'dummy'}
	g:add_event(lt.loss{name = 'dummy_loss'})
	g:initialize{cur_ver_dir = output_dir}

	local iter_count = 10

	for i = 1, iter_count do
		g.event_logger.epoch = 1
		g.event_logger.iteration = i
		g['dummy_loss']:update({epoch = 1, iteration = i, value = iter_count + 1 - i})
		if i ~= iter_count then g:flush_updates() end
	end

	local improved_metrics = g:summarize()
	g:flush_updates()
	tester:assert(improved_metrics.dummy_loss[1] == 'mean')

	local proto_file = io.open(paths.concat(output_dir, 'dummy/event_record.pb2'), 'rb')
	local def_str = proto_file:read('*a')
	proto_file:close()

	local def = lt.pb.load_proto(def_str)
	local record = def.Wrapper.EventRecord()

	local data = lt.load_binary(paths.concat(output_dir, 'dummy/event_data.dat'))
	local off = 1
	local count = 1

	while off <= string.len(data) do
		-- Ensure that the byte at the current offset indicates that we are at the start of
		-- a new `Wrapper` message, and extract its length.
		local wtype = bit.band(string.byte(data, off), 0x7)
		local len = string.byte(data, off + 1)
		assert(wtype == 2, "Expected start of new message.")

		local payload = string.sub(data, off + 2, off + 1 + len)
		record:Clear()
		record:Parse(payload)

		tester:assert(record.epoch == 1)
		tester:assert(record.iteration == count)
		tester:assert(record.dummy_loss.value == iter_count + 1 - count)

		if count ~= iter_count then
			tester:assert(record.dummy_loss.mean == nil)
		else
			tester:assert(record.dummy_loss.mean == 5.5)
		end

		off = off + 2 + len
		count = count + 1
	end

	lt.remove_directory(test_dir)
end

function tests.test_serialization()
	local test_dir = 'scratch/dummy_experiment'
	local output_dir = paths.concat(test_dir, 'current')

	lt.make_directory_if_not_exists(test_dir)
	lt.make_directory_if_not_exists(output_dir)

	local g1 = lt.event_group{name = 'dummy'}
	for i = 1, 3 do g1:add_event(lt.loss{name = F'loss_{i}'}) end

	local test_file = paths.concat(output_dir, 'test.t7')
	torch.save(test_file, g1)

	local g2 = torch.load(test_file)
	for i = 1, 3 do tester:assert(g2.state.name_to_event[F'loss_{i}'] ~= nil) end
	lt.remove_directory(test_dir)
end

return tester:add(tests):run()
