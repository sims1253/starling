import type { E2EConfig } from 'e2e';
import { mobile } from '@e2e-dev/mobile';
import { packageId } from './support.ts';

export default {
  targets: [{
    name: 'android',
    engine: mobile({ platform: 'android', device: process.env.ANDROID_SERIAL }),
    app: {
      bundleId: packageId,
      permissions: { microphone: 'grant', notifications: 'grant' },
    },
  }],
  workers: 1,
  retries: 0,
  timeout: 600_000,
  actionTimeout: 60_000,
  assertionTimeout: 120_000,
  cleanupTimeout: 60_000,
  trace: 'on',
  video: 'on',
  cache: 'off',
  output: 'artifacts/runner',
} satisfies E2EConfig;
