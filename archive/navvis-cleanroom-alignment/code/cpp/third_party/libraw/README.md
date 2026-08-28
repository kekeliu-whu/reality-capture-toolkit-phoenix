# LibRaw public headers

This directory contains the public headers from the Ubuntu LibRaw 0.20
development package. They allow the panorama worker to compile on a host that
has the ABI-compatible `libraw.so.20` runtime but lacks the unversioned
development symlink and headers. CMake prefers a system development package
when one is present.

LibRaw is a third-party project and is not part of the clean-room NavVis
implementation. Its original `COPYRIGHT`, `LICENSE.LGPL`, and `LICENSE.CDDL`
files are included alongside the headers.
