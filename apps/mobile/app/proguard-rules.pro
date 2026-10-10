# Starling Mobile currently uses only platform APIs. Keep this file so a
# future embedded engine or optional transport can add release rules locally.

# The JNI bridge calls StarlingNative.Cancel.requested() by name.
-keep interface dev.starling.mobile.engine.StarlingNative$Cancel { boolean requested(); }
