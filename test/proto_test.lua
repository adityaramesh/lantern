require('lt')
require('totem')

local tests = totem.TestSuite()
local tester = totem.Tester()

function tests.test_proto_generation()
	local s1 = 
[[message Test {
	optional int32 f_1 = 1;
}]]

	local s2 = 
[[message Test {
	optional int32 f_1 = 1;
	required int32 f_2 = 2;
	optional int32 f_3 = 3;
}]]

	local s3 = 
[[message Test_1 {
	message Test_2 {
		optional int32 f_1 = 1;
		required int32 f_2 = 2;
		optional int32 f_3 = 3;
	}
	message Test_3 {
		optional int32 f_1 = 1;
		required int32 f_2 = 2;
		optional int32 f_3 = 3;
	}
	optional int32 f_1 = 1;
	optional Test_2 m_1 = 2;
	required Test_3 m_2 = 3;
}]]

	local make_definitions = function()
		local m1 = lt.message_definition({def_name = 'Test'})
		m1:add_field(lt.field_definition{name = 'f_1', type = 'int32'})

		local m2 = lt.message_definition({def_name = 'Test'})
		m2:add_field(lt.field_definition{name = 'f_1', type = 'int32'})
		m2:add_field(lt.field_definition{name = 'f_2', type = 'int32', rule = 'required'})
		m2:add_field(lt.field_definition{name = 'f_3', type = 'int32'})

		local m3 = lt.message_definition({def_name = 'Test_1'})
		m3:add_field(lt.field_definition{name = 'f_1', type = 'int32'})

		local m3_1 = lt.message_definition({def_name = 'Test_2', name = 'm_1'})
		m3_1:add_field(lt.field_definition{name = 'f_1', type = 'int32'})
		m3_1:add_field(lt.field_definition{name = 'f_2', type = 'int32', rule = 'required'})
		m3_1:add_field(lt.field_definition{name = 'f_3', type = 'int32'})

		local m3_2 = lt.message_definition({def_name = 'Test_3', name = 'm_2',
			rule = 'required'})
		m3_2:add_field(lt.field_definition{name = 'f_1', type = 'int32'})
		m3_2:add_field(lt.field_definition{name = 'f_2', type = 'int32', rule = 'required'})
		m3_2:add_field(lt.field_definition{name = 'f_3', type = 'int32'})

		m3:add_message(m3_1)
		m3:add_message(m3_2)

		return m1, m2, m3
	end

	local m1, m2, m3 = make_definitions()
	print(m1:to_proto())
	tester:assert(m1:to_proto() == s1)
	tester:assert(m2:to_proto() == s2)
	tester:assert(m3:to_proto() == s3)

	local n1, n2, n3 = make_definitions()
	tester:assert(m1 ~= n2)
	tester:assert(m1 ~= n3)
	tester:assert(m2 ~= n3)

	tester:assert(m1 == n1)
	tester:assert(m2 == n2)
	tester:assert(m3 == n3)
end

return tester:add(tests):run()
