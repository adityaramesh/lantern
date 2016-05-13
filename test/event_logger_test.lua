require('lt')
require('totem')

local tests = totem.TestSuite()
local tester = totem.Tester()

function tests.test_event_logger()
end

return tester:add(tests):run()
