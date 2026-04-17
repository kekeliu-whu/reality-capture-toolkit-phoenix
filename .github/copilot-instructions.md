# Copilot Instructions

## C++ Code Style

All C++ code in this project must follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Key rules:

- **Naming**:
  - Files: `snake_case.cc` / `snake_case.h`
  - Types (classes, structs, enums, type aliases): `CamelCase`
  - Variables: `snake_case`, class data members with trailing underscore `member_var_`
  - Constants: `kCamelCase`
  - Functions: `CamelCase`
  - Namespaces: `snake_case`
  - Enumerators: `kCamelCase`
  - Macros: `UPPER_SNAKE_CASE`
- **Formatting**:
  - Indent with 2 spaces, no tabs
  - Opening brace on the same line as the statement
  - Max line length: 80 characters
  - Use `#pragma once` or include guards (`#ifndef PROJECT_PATH_FILE_H_`)
- **Headers**:
  - Include order: related header, C system headers, C++ standard library headers, other library headers, project headers — each group separated by a blank line
  - Prefer forward declarations to avoid unnecessary includes
- **Language features**:
  - Use `nullptr` instead of `NULL` or `0`
  - Use `const` and `constexpr` where appropriate
  - Prefer `std::unique_ptr` / `std::shared_ptr` over raw owning pointers
  - Use `auto` when the type is obvious or the full type is long
  - Prefer range-based `for` loops
  - Use `override` and `final` for virtual methods
  - Use scoped enums (`enum class`) over plain enums
- **Comments**: Use `//` for single-line comments; `/** ... */` style for documentation if needed
- **Namespaces**: Do not use `using namespace std;` in headers; prefer explicit qualifications

## Python Code Style

- **禁止隐式环境适配 monkey-patch**：不要为了兼容模块在不同路径下被 import 两次的情况，写类似下面的 patch 代码：
  ```python
  # ❌ 不要这样写
  try:
      from some.fully.qualified.module import Foo as _Foo
      _Foo.forward = _patched_forward
  except ImportError:
      pass
  ```
  这种写法会隐藏真正的导入/路径问题，出 bug 极难排查。除非用户明确要求适配此场景，否则一律不写。
  如果遇到同一个模块被 Python 解析为两个不同 module object 的问题，应该修正根因（`sys.path`、包结构、`__init__.py`），而不是用 monkey-patch 绕过。

## 环境配置文档化

如果 AI 在完成任务过程中做了较复杂的环境配置（如安装依赖、编译扩展、配置路径、修改系统设置等），**必须**将配置步骤和注意事项写入 `AGENTS.md`。如果不确定该写在 `AGENTS.md` 的哪个位置，先问用户。

## 常用命令

### run-ar（点云重建流水线）
```powershell
.\ztools\run-ar.ps1 `
  -inputdir D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud `
  -insvpath D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud\VID_20260415_122738_00_228.insv `
  -outputdir D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud\output `
  -calibfile D:\output\calibration.dat `
  -movpath "D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud\2026-04-16 113900.mov" `
  -trajectoryfile "D:\Users\rick\Desktop\2026-04-15_12-26-00_PointCloud\Project_2026-04-16_14-49-15\2026-04-15_12-26-00_PointCloud\output\trajectory.txt"
```
