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
