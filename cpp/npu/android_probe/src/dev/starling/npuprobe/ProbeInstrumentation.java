// Copyright 2026 The Starling Authors. Apache-2.0.
package dev.starling.npuprobe;

import android.app.Activity;
import android.app.Instrumentation;
import android.os.Bundle;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * `am instrument -w -e step compile|run dev.starling.npuprobe/.ProbeInstrumentation`
 *
 * Runs in the probe app's own process (untrusted_app domain, app linker
 * namespace). Inputs live in the app's external files dir (adb-pushable); the
 * compiled artifact goes to the app's private files dir, as a real app would
 * cache it.
 */
public final class ProbeInstrumentation extends Instrumentation {
    private Bundle args;

    @Override
    public void onCreate(Bundle arguments) {
        args = arguments;
        start();
    }

    @Override
    public void onStart() {
        Bundle out = new Bundle();
        StringBuilder r = new StringBuilder();
        try {
            r.append("selinux ").append(new String(Files.readAllBytes(Paths.get("/proc/self/attr/current"))).trim())
                    .append(", uid ").append(android.os.Process.myUid()).append('\n');
            File ext = getContext().getExternalFilesDir(null);
            // -e art ext: use an artifact pushed to the external files dir (compiled elsewhere).
            File art = "ext".equals(args.getString("art")) ? new File(getContext().getExternalFilesDir(null), "encoder.stnpu")
                    : new File(getContext().getFilesDir(), "encoder.stnpu");
            String step = args.getString("step", "run");
            long t0 = System.nanoTime();
            if (step.equals("compile")) {
                r.append(NpuProbe.compile(new File(ext, args.getString("model", "model.tflite")).getPath(),
                        art.getPath(), args.getString("fn", "subgraph_0_fn"), new long[] {409600, 416}, new long[] {425984}));
            } else {
                r.append(NpuProbe.run(art.getPath(), new File(ext, "mel_w800.f32").getPath(),
                        new File(ext, "kbias_w800.f32").getPath(), new File(ext, "app").getPath(),
                        Integer.parseInt(args.getString("runs", "20"))));
            }
            r.append(String.format("step '%s' wall incl. JNI %.1f ms\n", step, (System.nanoTime() - t0) / 1e6));
            out.putString("report", r.toString());
            finish(Activity.RESULT_OK, out);
        } catch (Throwable t) {
            out.putString("report", r + "EXCEPTION " + t);
            finish(Activity.RESULT_CANCELED, out);
        }
    }
}
