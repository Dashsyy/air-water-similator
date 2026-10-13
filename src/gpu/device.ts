export interface GpuContext {
  device: GPUDevice;
  adapter: GPUAdapter;
  maxCells: number;
}

/** Bytes of one distribution buffer per lattice cell (19 float32 populations). */
export const BYTES_PER_CELL = 19 * 4;

export async function initGpu(): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new Error('WebGPU is not available in this browser. Use a recent Chrome, Edge, or Safari.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter was found on this device.');

  const wantedBinding = Math.min(adapter.limits.maxStorageBufferBindingSize, 1024 * 1024 * 1024);
  const wantedBuffer = Math.min(adapter.limits.maxBufferSize, 1024 * 1024 * 1024);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: wantedBinding,
      maxBufferSize: wantedBuffer,
    },
  });
  const maxCells = Math.floor(Math.min(wantedBinding, wantedBuffer) / BYTES_PER_CELL);
  return { device, adapter, maxCells };
}
