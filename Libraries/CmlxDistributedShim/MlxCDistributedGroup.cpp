// Compile the pinned ABI implementation here; the Cmlx target excludes it.
// Do not shadow upstream signatures: 0.32 uses output parameters and exposes
// explicit group creation/freeing.
#include "mlx/c/distributed_group.cpp"
