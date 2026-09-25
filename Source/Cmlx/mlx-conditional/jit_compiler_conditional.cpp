// Copyright © 2025 Apple Inc.

#include <TargetConditionals.h>

// Keep this condition in sync with compiled_conditional.cpp. CPU JIT shells
// out through std::system(), which is unavailable on iOS and visionOS.
#if !(TARGET_OS_IOS || TARGET_OS_VISION)
#include "../mlx/mlx/backend/cpu/jit_compiler.cpp"
#endif
