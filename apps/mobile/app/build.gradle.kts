plugins {
    id("com.android.application")
}

android {
    namespace = "dev.starling.mobile"
    compileSdk = 35

    defaultConfig {
        applicationId = "dev.starling.mobile"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    ndkVersion = "28.2.13676358"

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
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
