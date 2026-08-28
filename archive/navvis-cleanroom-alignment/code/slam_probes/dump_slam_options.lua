include "slam_config_offline.lua"

-- Read-only diagnostic config used with surveyor_ros/check_config.  The
-- installed Lua loader exposes the fully merged option table here even though
-- the distributed source files are encrypted.  Keep the output stable so it
-- can be diffed between installed builds.
local function dump_table(prefix, value)
  if type(value) ~= "table" then
    print(prefix .. "=" .. tostring(value))
    return
  end
  local keys = {}
  for key, _ in pairs(value) do
    table.insert(keys, key)
  end
  table.sort(keys, function(left, right)
    return tostring(left) < tostring(right)
  end)
  for _, key in ipairs(keys) do
    local child = value[key]
    local child_prefix = prefix .. "." .. tostring(key)
    if type(child) == "table" then
      dump_table(child_prefix, child)
    elseif type(child) == "number" or type(child) == "boolean" or
        type(child) == "string" then
      print(child_prefix .. "=" .. tostring(child))
    end
  end
end

dump_table("options", options)
return options
