--[[
Defines common filesystem utility functions with built-in error checking.
--]]

local F   = lt.F
local os  = lt.os
local lfs = lt.lfs

function lt.remove_file(path, logger)
	local success, err = os.remove(path)
	lt.fail_if(not success, F"Failed to remove file '{path}': {err}", logger)
end

function lt.remove_file_if_exists(path, logger)
	if not paths.filep(path) then return end
	lt.remove_file(path, logger)
end

function lt.rename_file(old_path, new_path, logger)
	lt.fail_if(paths.filep(new_path), F"Path '{new_path}' already exists.", logger)
	local success, err = os.rename(old_path, new_path)
	lt.fail_if(not success, F"Failed to rename '{old_path}' to '{new_path}': {err}",
		logger)
end

function lt.rename_file_if_exists(old_path, new_path, logger)
	if not paths.filep(old_path) then return end
	lt.rename_file(old_path, new_path, logger)
end

function lt.create_hard_link(src_path, dst_path, logger)
	local success, err = lfs.link(src_path, dst_path)
	lt.fail_if(not success, F"Failed creating hard link '{dst_path}': {err}", logger)
end

function lt.make_directory(dir, logger)
	local success, err = lfs.mkdir(dir)
	lt.fail_if(not success, F"Failed to make directory '{dir}': {err}", logger)
end

function lt.make_directory_if_not_exists(dir, logger)
	if paths.dirp(dir) then return end
	lt.make_directory(dir, logger)
end

function lt.remove_empty_directory(dir, logger)
	local success, err = lfs.rmdir(dir)
	lt.fail_if(not success, F"Failed to remove directory '{dir}': {err}", logger)
end

function lt.remove_directory(dir, logger)
	for entry in lfs.dir(dir) do
		if entry == '.' or entry == '..' then goto continue end
		local path = paths.concat(dir, entry)
		local mode = lfs.attributes(path, 'mode')

		if mode ~= 'directory' then
			lt.remove_file(path)
		else
			lt.remove_directory(path)
		end

		::continue::
	end

	lt.remove_empty_directory(dir)
end

function lt.remove_directory_if_exists(dir, logger)
	if not paths.dirp(dir) then return end
	lt.remove_directory(dir, logger)
end

--[[
Checks if both paths refer to the same underlying file by following all links.
--]]
function lt.is_same_file(src_path, dst_path, logger)
	local src_info, src_err = sys_stat.stat(src_path)
	lt.fail_if(src_info == nil, F"Failed to stat file '{src_path}': {src_err}.", logger)

	local dst_info, dst_err = sys_stat.stat(dst_path)
	lt.fail_if(dst_info == nil, F"Failed to stat file '{dst_path}': {dst_err}.", logger)

	return src_info.st_dev == dst_info.st_dev and src_info.st_ino == dst_info.st_ino
end
