#ifndef VMLX_MAPPED_FILE_H
#define VMLX_MAPPED_FILE_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

// Creates a read-only virtual mapping, not a Metal buffer. The caller supplies
// a verified regular-file descriptor and an initially null owned handle.
int vmlx_mapped_file_create(int32_t fd, void **owner);
void vmlx_mapped_file_release(void **owner);
// Each array owns its small Metal resource and shares the mapping's lifetime.
// The file handle must remain alive until this call returns, but may then be
// released before the array. Shape/offset/source identity are checked.
int vmlx_mapped_file_array(void **array_context, void *owner, uint64_t offset,
                           size_t length, const int32_t *shape, int32_t rank,
                           int32_t dtype);
#ifdef __cplusplus
}
#endif
#endif
