plugins {
    id("com.android.application")
}

// Release builds (the release-android workflow) pass the tag-derived version
// as Gradle properties; local builds keep the defaults below.
val starlingVersionName = providers.gradleProperty("starlingVersionName").orElse("0.1.0").get()
val starlingVersionCode = providers.gradleProperty("starlingVersionCode").orElse("1").get().let { raw ->
    raw.toIntOrNull()?.takeIf { it > 0 }
        ?: throw GradleException("starlingVersionCode must be an integer in 1..${Int.MAX_VALUE}: $raw")
}

// arm64 codegen target of the native engine. The default runs on every
// mainstream arm64 phone core since 2018; `-PstarlingArmArch=armv8.2-a+dotprod+fp16+i8mm`
// builds the variant with int8 matrix-multiply kernels (Tensor G3+/Pixel 8+,
// Snapdragon 8 Gen 1+), which the release workflow publishes as a separate APK.
// Each extension the target enables becomes a CPU feature NativeSupport
// requires at runtime, so a phone without it gets an explanation, not SIGILL.
val starlingArmArch = providers.gradleProperty("starlingArmArch").orElse("armv8.2-a+dotprod+fp16").get()
val cpuFeatureNames = mapOf("dotprod" to "asimddp", "fp16" to "asimdhp", "i8mm" to "i8mm")
val starlingArmExtensions = starlingArmArch.split('+').drop(1)
require(starlingArmExtensions.none(String::isBlank)) {
    "starlingArmArch has an empty '+' extension (malformed): $starlingArmArch"
}
require(starlingArmExtensions.all { it in cpuFeatureNames }) {
    "starlingArmArch may only enable ${cpuFeatureNames.keys}: $starlingArmArch"
}
val requiredCpuFeatures = starlingArmExtensions.map(cpuFeatureNames::getValue)

// Optional Vulkan GPU backend for the on-device engine (`-PstarlingVulkan=true`).
// The release workflow enables it for the i8mm APK only (recent phones); the
// app keeps the CPU as the default device and offers the GPU as an opt-in.
val starlingVulkan = providers.gradleProperty("starlingVulkan").orElse("false").get().toBooleanStrict()

// Release signing is configured only when all four variables are present, so
// `assembleRelease` without them still works and produces an unsigned APK.
// The keystore must stay the same across releases: Android refuses to update
// an installed app with an APK signed by a different key.
val signingEnv = listOf(
    "STARLING_ANDROID_KEYSTORE",
    "STARLING_ANDROID_KEYSTORE_PASSWORD",
    "STARLING_ANDROID_KEY_ALIAS",
    "STARLING_ANDROID_KEY_PASSWORD",
).associateWith { providers.environmentVariable(it).orNull?.takeIf(String::isNotEmpty) }
val missingSigningEnv = signingEnv.filterValues { it == null }.keys
// Some but not all set is a misconfiguration, not "no signing": say which.
if (missingSigningEnv.isNotEmpty() && missingSigningEnv.size < signingEnv.size) {
    throw GradleException("Release signing is partially configured; also set: ${missingSigningEnv.joinToString()}")
}
val releaseSigning = signingEnv.takeIf { missingSigningEnv.isEmpty() }?.mapValues { requireNotNull(it.value) }

// A bad keystore path only matters when a release build will sign with it;
// checked once the task graph is known so debug builds are never blocked.
if (releaseSigning != null) {
    val keystore = file(releaseSigning.getValue("STARLING_ANDROID_KEYSTORE"))
    gradle.taskGraph.whenReady {
        if (allTasks.any { it.name.contains("Release") } && !keystore.isFile) {
            throw GradleException("STARLING_ANDROID_KEYSTORE does not exist: $keystore")
        }
    }
}

android {
    namespace = "dev.starling.mobile"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.starling.mobile"
        // ggml-vulkan links four Vulkan 1.1 entry points directly
        // (vkGetPhysicalDeviceFeatures2, ...), which libvulkan exports from
        // API 28 on; every phone the Vulkan (i8mm) APK targets runs Android 12+.
        minSdk = if (starlingVulkan) 28 else 26
        targetSdk = 35
        versionCode = starlingVersionCode
        versionName = starlingVersionName
        buildConfigField("String", "ARM64_ARCH", "\"$starlingArmArch\"")
        buildConfigField("boolean", "VULKAN_BUILD", starlingVulkan.toString())
        buildConfigField(
            "String[]",
            "ARM64_REQUIRED_CPU_FEATURES",
            requiredCpuFeatures.joinToString(prefix = "{", postfix = "}") { "\"$it\"" },
        )
        externalNativeBuild {
            cmake {
                arguments += "-DSTARLING_ANDROID_ARM_ARCH=$starlingArmArch"
                arguments += "-DSTARLING_ANDROID_VULKAN=${if (starlingVulkan) "ON" else "OFF"}"
            }
        }
        ndk {
            // The Vulkan backend embeds ~37 MB of SPIR-V per ABI; its APK
            // targets arm64 phones only (emulators use the standard APK).
            abiFilters += if (starlingVulkan) listOf("arm64-v8a") else listOf("arm64-v8a", "x86_64")
        }
    }

    ndkVersion = "28.2.13676358"

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.1"
        }
    }

    signingConfigs {
        if (releaseSigning != null) {
            create("release") {
                storeFile = file(releaseSigning.getValue("STARLING_ANDROID_KEYSTORE"))
                storePassword = releaseSigning.getValue("STARLING_ANDROID_KEYSTORE_PASSWORD")
                keyAlias = releaseSigning.getValue("STARLING_ANDROID_KEY_ALIAS")
                keyPassword = releaseSigning.getValue("STARLING_ANDROID_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            if (releaseSigning != null) signingConfig = signingConfigs.getByName("release")
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

dependencies {
    implementation("androidx.core:core:1.16.0")
    // WebSocket client for the /stream live-dictation protocol; the batch
    // upload path stays on HttpURLConnection. OkHttp 4.x is used because
    // the 5.x Android artifact requires a newer compileSdk than this app;
    // MockWebServer is pinned to the same line so tests exercise one
    // implementation, not two stitched versions.
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20260814")
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
}
