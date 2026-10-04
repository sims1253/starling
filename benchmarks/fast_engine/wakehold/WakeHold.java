// WakeHold — hold a PARTIAL_WAKE_LOCK as the shell uid while it runs (#325).
//
//   adb shell CLASSPATH=/data/local/tmp/starling/wakehold.dex \
//     app_process / WakeHold [tag] [max_seconds]
//
// GPU work on a phone that may suspend wedges the PowerVR driver (RESEARCH_LOG
// P3-3), and the measurement scripts need the screen off. Shell cannot write
// /sys/power/wake_lock on a user build, but it holds android.permission.
// WAKE_LOCK, and a wake lock of a uid below FIRST_APPLICATION_UID is not
// disabled by Doze. The lock token is a Binder owned by this process:
// PowerManagerService releases the lock when the process dies, so killing it
// (TERM or KILL) always releases — there is no release path to get wrong.
// max_seconds (default 7200) bounds a holder whose owner forgot it.
//
// IPowerManager.acquireWakeLock's signature changes across releases, so the
// call is resolved by reflection and its arguments filled by type.
import android.os.Binder;
import android.os.IBinder;
import java.lang.reflect.Method;

public class WakeHold {
    public static void main(String[] args) throws Exception {
        String tag = args.length > 0 ? args[0] : "starling-bench";
        long maxSeconds = args.length > 1 ? Long.parseLong(args[1]) : 7200;

        IBinder service = (IBinder) Class.forName("android.os.ServiceManager")
                .getMethod("getService", String.class).invoke(null, "power");
        Object pm = Class.forName("android.os.IPowerManager$Stub")
                .getMethod("asInterface", IBinder.class).invoke(null, service);

        Method acquire = null;
        for (Method m : pm.getClass().getMethods())
            if (m.getName().equals("acquireWakeLock")
                    && (acquire == null || m.getParameterCount() > acquire.getParameterCount()))
                acquire = m;
        if (acquire == null) throw new IllegalStateException("no IPowerManager.acquireWakeLock");

        // Known orders: (IBinder lock, int flags, String tag, String packageName,
        // WorkSource ws, String historyTag[, int displayId[, IWakeLockCallback cb]]).
        Class<?>[] types = acquire.getParameterTypes();
        Object[] a = new Object[types.length];
        Binder token = new Binder();
        int ints = 0, strings = 0;
        for (int i = 0; i < types.length; i++) {
            Class<?> t = types[i];
            if (t == IBinder.class) a[i] = token;
            else if (t == int.class) a[i] = ints++ == 0 ? 1 /* PARTIAL_WAKE_LOCK */ : -1 /* INVALID_DISPLAY */;
            else if (t == String.class) {
                a[i] = strings == 0 ? tag : strings == 1 ? "com.android.shell" : null;
                strings++;
            } else if (t == boolean.class) a[i] = false;
            else a[i] = null;   // WorkSource, IWakeLockCallback
        }
        acquire.invoke(pm, a);
        System.out.println("holding " + tag + " (pid " + android.os.Process.myPid() + ")");
        System.out.flush();
        Thread.sleep(maxSeconds * 1000);
    }
}
