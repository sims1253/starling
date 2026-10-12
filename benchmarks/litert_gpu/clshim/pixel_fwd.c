// libOpenCL-pixel.so stand-in: the Pixel loader interface (enableOpenCL /
// loadOpenCLPointer) resolved against clshim's libOpenCL.so, so an accelerator
// that takes the Pixel route still goes through the shim.
#include <dlfcn.h>
#include <stdio.h>

static void* shim(void) {
  static void* h;
  if (!h) h = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
  return h;
}
void enableOpenCL(void) { (void)shim(); }
void* loadOpenCLPointer(const char* name) {
  void* h = shim();
  return h ? dlsym(h, name) : NULL;
}
