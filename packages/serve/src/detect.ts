/**
 * Machine inspection, isolated from the pure mapping logic so tests can inject
 * deterministic answers.
 */
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/** Common loader locations for x86_64 distributions. */
const VULKAN_LOADER_PATHS = [
  "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
  "/lib/x86_64-linux-gnu/libvulkan.so.1",
  "/usr/lib64/libvulkan.so.1",
  "/usr/lib/libvulkan.so.1",
] as const;

/**
 * Best-effort detection of the Vulkan loader (`libvulkan.so.1`) on Linux.
 * Only the presence of the loader is probed; whether the installed driver can
 * actually run inference is the executable's concern, not the launcher's.
 */
export function detectVulkanLoader(): boolean {
  if (process.platform !== "linux") return false;

  for (const path of VULKAN_LOADER_PATHS) {
    if (existsSync(path)) return true;
  }

  return false;
}

/** Ask the dynamic linker cache about the loader; wrapped for tests. */
export async function detectVulkanLoaderViaLdconfig(): Promise<boolean> {
  if (process.platform !== "linux") return false;

  try {
    const { stdout } = await execFileAsync("ldconfig", ["-p"]);

    return /libvulkan\.so\.1/i.test(stdout);
  } catch {
    return false;
  }
}
