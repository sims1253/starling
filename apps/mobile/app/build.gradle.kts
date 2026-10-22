plugins {
    id("com.android.application")
}

// Release builds (the release-android workflow) pass the tag-derived version
// as Gradle properties; local builds keep the defaults below.
val starlingVersionName = providers.gradleProperty("starlingVersionName").orElse("0.1.0").get()
fun gradleFlag(name: String): Boolean {
    val raw = providers.gradleProperty(name).orElse("false").get()
    return raw.toBooleanStrictOrNull()
        ?: throw GradleException("$name must be 'true' or 'false': $raw")
}
val starlingExperimental = gradleFlag("starlingExperimental")
val starlingArm64Only = gradleFlag("starlingArm64Only")
// CI shares native compilation through ccache; local builds keep the normal
// compiler invocation unless a launcher is explicitly requested.
val starlingCompilerLauncher = providers.gradleProperty("starlingCompilerLauncher").orElse("").get()
// Builds the debug APK's native engine optimized (RelWithDebInfo), so
// on-device tests of the larger models finish in seconds, not minutes.
val starlingOptimizedDebugNative = gradleFlag("starlingOptimizedDebugNative")
val starlingVersionCode = providers.gradleProperty("starlingVersionCode").orElse("1").get().let { raw ->
    raw.toIntOrNull()?.takeIf { it > 0 }
        ?: throw GradleException("starlingVersionCode must be an integer in 1..${Int.MAX_VALUE}: $raw")
}
// The commit a debug APK is built from, so on-device measurements (#357)
// record which build they ran; "unknown" outside a git checkout.
val starlingGitSha = runCatching {
    providers.exec {
        commandLine("git", "rev-parse", "HEAD")
        isIgnoreExitValue = true
    }.standardOutput.asText.get().trim()
}.getOrNull()?.takeIf { Regex("[0-9a-f]{40}").matches(it) } ?: "unknown"

// arm64 codegen target of the native engine. The default runs on every
// mainstream arm64 phone core since 2018; `-PstarlingArmArch=armv8.2-a+dotprod+fp16+i8mm`
// builds the variant with int8 matrix-multiply kernels (Tensor G3+/Pixel 8+,
// Snapdragon 8 Gen 1+), which the release workflow publishes as a separate APK.
// Each extension the target enables becomes a CPU feature NativeSupport
// requires at runtime, so a phone without it gets an explanation, not SIGILL.
val starlingArmArch = providers.gradleProperty("starlingArmArch").orElse("armv8.2-a+dotprod+fp16").get()
val cpuFeatureNames = mapOf("dotprod" to "asimddp", "fp16" to "asimdhp", "i8mm" to "i8mm")
val starlingArmExtensions = starlingArmArch.split('+').drop(1)
require(Regex("armv[89]\\.[0-9]+-a").matches(starlingArmArch.substringBefore('+'))) {
    "starlingArmArch must start with an arm64 base such as armv8.2-a: $starlingArmArch"
}
require(starlingArmExtensions.none(String::isBlank)) {
    "starlingArmArch has an empty '+' extension (malformed): $starlingArmArch"
}
require(starlingArmExtensions.all { it in cpuFeatureNames }) {
    "starlingArmArch may only enable ${cpuFeatureNames.keys}: $starlingArmArch"
}
val requiredCpuFeatures = starlingArmExtensions.map(cpuFeatureNames::getValue)

// CMake build type of the native engine in every variant. AGP's default is
// Debug for debuggable variants, which compiles ggml at -O0, yet the debug
// APK is what on-device tests and measurements run; this matches the release
// variant's AGP default (RelWithDebInfo) instead. `-PstarlingNativeBuildType=Debug`
// restores an unoptimized engine for native debugging.
val starlingNativeBuildType = providers.gradleProperty("starlingNativeBuildType").orElse("RelWithDebInfo").get()
require(starlingNativeBuildType in setOf("Debug", "Release", "RelWithDebInfo", "MinSizeRel")) {
    "starlingNativeBuildType must be a CMake build type: $starlingNativeBuildType"
}

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
        if (allTasks.any { it.project == project && it.name.contains("Release") } && !keystore.isFile) {
            throw GradleException("STARLING_ANDROID_KEYSTORE does not exist: $keystore")
        }
    }
}

// Contract data shared with the Python oracles and the Rust ports
// (packages/contracts/mode-routing, insertion-boundary). The JVM tests replay its fixtures in
// place; the runtime tables ship as assets, copied at build time, so the
// app and every other port read one copy.
val contractsDir = layout.projectDirectory.dir("../../../packages/contracts")

@CacheableTask
abstract class ContractAssetsTask : DefaultTask() {
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.NAME_ONLY)
    abstract val tables: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val out = outputDir.get().asFile.resolve("contracts")
        out.deleteRecursively()
        check(out.mkdirs() || out.isDirectory) { "Unable to create the contract assets directory $out" }
        tables.files.forEach { it.copyTo(out.resolve(it.name)) }
    }
}

val contractAssets = tasks.register<ContractAssetsTask>("contractAssets") {
    tables.from(
        contractsDir.file("mode-routing/spoken-commands.json"),
        contractsDir.file("mode-routing/spoken-instructions.json"),
    )
}

// Third-party notices (licenses/README.md): every runtime Maven artifact of
// the variant, licensed by the rules in licenses/android-dependencies.json,
// plus the engine's C/C++ components and fonts from licenses/components.json,
// as assets/licenses/third-party-notices.json for the Licenses screen. The
// same schema scripts/licenses/notices.py writes for the desktop app.
val repoRoot = layout.projectDirectory.dir("../../..")

abstract class ThirdPartyNoticesTask : DefaultTask() {
    /** group:artifact:version of every runtime module. */
    @get:Input
    abstract val modules: ListProperty<String>

    /** licenses/, the repository LICENSE and the license files components.json names. */
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val licenseFiles: ConfigurableFileCollection

    @get:Internal
    abstract val root: DirectoryProperty

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun generate() {
        val root = root.get().asFile
        val texts = sortedMapOf<String, String>()
        val missing = mutableListOf<String>()
        fun addText(text: String): String {
            val digest = java.security.MessageDigest.getInstance("SHA-256").digest(text.toByteArray())
            val key = digest.joinToString("") { "%02x".format(it) }.take(16)
            texts.putIfAbsent(key, text)
            return key
        }
        fun read(file: File): String =
            file.readText().replace("\r\n", "\n").lines().joinToString("\n") { it.trimEnd() }.trim() + "\n"
        fun text(spec: String, who: String): String? {
            val file = if (spec.startsWith("spdx:")) root.resolve("licenses/spdx/${spec.removePrefix("spdx:")}.txt")
            else root.resolve(spec)
            if (!file.isFile) {
                missing += "$who: $spec does not exist"
                return null
            }
            return addText(read(file))
        }

        @Suppress("UNCHECKED_CAST")
        fun json(path: String) = groovy.json.JsonSlurper().parse(root.resolve(path)) as Map<String, Any?>
        @Suppress("UNCHECKED_CAST")
        val rules = json("licenses/android-dependencies.json")["rules"] as List<Map<String, Any?>>
        val components = mutableListOf<Map<String, Any?>>()
        for (module in modules.get()) {
            val (group, name, version) = module.split(":")
            val rule = rules.firstOrNull { "$group:$name".startsWith(it["match"] as String) }
            if (rule == null) {
                missing += "$module: no license rule; check its license and add one to licenses/android-dependencies.json"
                continue
            }
            @Suppress("UNCHECKED_CAST")
            val keys = (rule["texts"] as List<String>).mapNotNull { text(it, module) }
            components += mapOf(
                "name" to "$group:$name", "version" to version, "license" to rule["license"],
                "url" to rule["url"], "texts" to keys,
            )
        }
        @Suppress("UNCHECKED_CAST")
        for (component in json("licenses/components.json")["components"] as List<Map<String, Any?>>) {
            if ("android" !in (component["apps"] as List<*>)) continue
            @Suppress("UNCHECKED_CAST")
            val keys = (component["texts"] as List<String>).mapNotNull { text(it, component["name"] as String) }
            components += component.filterKeys { it in setOf("name", "version", "license", "url") } + ("texts" to keys)
        }
        if (missing.isNotEmpty()) {
            throw GradleException("Third-party notices are incomplete:\n  " + missing.joinToString("\n  "))
        }
        val notices = mapOf(
            "schema" to 1,
            "app" to "android",
            "starling" to mapOf("license" to "MIT", "text" to addText(read(root.resolve("LICENSE")))),
            "components" to components.sortedWith(compareBy({ (it["name"] as String).lowercase() }, { it["version"] as String })),
            "texts" to texts,
        )
        val out = outputDir.get().asFile.resolve("licenses")
        out.deleteRecursively()
        check(out.mkdirs() || out.isDirectory) { "Unable to create the notices assets directory $out" }
        out.resolve("third-party-notices.json").writeText(groovy.json.JsonOutput.toJson(notices))
    }
}

/** Every external module in a resolved dependency graph, as group:artifact:version. */
fun runtimeModules(root: org.gradle.api.artifacts.result.ResolvedComponentResult): List<String> {
    val seen = mutableSetOf<org.gradle.api.artifacts.component.ComponentIdentifier>()
    val queue = ArrayDeque(listOf(root))
    val modules = sortedSetOf<String>()
    while (queue.isNotEmpty()) {
        val component = queue.removeFirst()
        if (!seen.add(component.id)) continue
        (component.id as? org.gradle.api.artifacts.component.ModuleComponentIdentifier)?.let {
            modules += "${it.group}:${it.module}:${it.version}"
        }
        component.dependencies.filterIsInstance<org.gradle.api.artifacts.result.ResolvedDependencyResult>()
            .forEach { queue += it.selected }
    }
    return modules.toList()
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(contractAssets, ContractAssetsTask::outputDir)
        val notices = tasks.register<ThirdPartyNoticesTask>("${variant.name}ThirdPartyNotices") {
            modules.set(variant.runtimeConfiguration.incoming.resolutionResult.rootComponent.map(::runtimeModules))
            root.set(repoRoot)
            licenseFiles.from(
                repoRoot.dir("licenses"),
                repoRoot.file("LICENSE"),
                repoRoot.file("third_party/ggml/LICENSE"),
                repoRoot.dir("apps/desktop-gpui/crates/app/assets/fonts").asFileTree.matching { include("*.txt") },
            )
        }
        variant.sources.assets?.addGeneratedSourceDirectory(notices, ThirdPartyNoticesTask::outputDir)
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("starling.contracts", contractsDir.asFile.absolutePath)
    inputs.dir(contractsDir.dir("mode-routing")).withPropertyName("contracts").withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.dir(contractsDir.dir("insertion-boundary")).withPropertyName("boundaryContract")
        .withPathSensitivity(PathSensitivity.RELATIVE)
}

android {
    namespace = "dev.starling.mobile"
    compileSdk = 37

    defaultConfig {
        applicationId = "dev.starling.mobile"
        minSdk = 26
        targetSdk = 35
        versionCode = starlingVersionCode
        versionName = starlingVersionName
        buildConfigField("String", "ARM64_ARCH", "\"$starlingArmArch\"")
        buildConfigField(
            "String[]",
            "ARM64_REQUIRED_CPU_FEATURES",
            requiredCpuFeatures.joinToString(prefix = "{", postfix = "}") { "\"$it\"" },
        )
        externalNativeBuild {
            cmake {
                arguments += "-DSTARLING_ANDROID_ARM_ARCH=$starlingArmArch"
                arguments += "-DCMAKE_BUILD_TYPE=$starlingNativeBuildType"
                if (starlingCompilerLauncher.isNotBlank()) {
                    arguments += "-DCMAKE_C_COMPILER_LAUNCHER=$starlingCompilerLauncher"
                    arguments += "-DCMAKE_CXX_COMPILER_LAUNCHER=$starlingCompilerLauncher"
                }
            }
        }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += if (starlingArm64Only) listOf("arm64-v8a") else listOf("arm64-v8a", "x86_64")
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

    // Experimental builds overlay the user-visible names. A release-source-set
    // overlay outranks src/main, whereas resValue would collide with the same
    // names in src/main/res/values/strings.xml.
    if (starlingExperimental) {
        sourceSets.getByName("release").res.srcDir("src/experimental/res")
    }

    buildTypes {
        // Installs beside a release build, so device tests never need to
        // uninstall (and wipe) the app someone actually uses.
        debug {
            applicationIdSuffix = ".debug"
            buildConfigField("String", "GIT_SHA", "\"$starlingGitSha\"")
            if (starlingOptimizedDebugNative) {
                externalNativeBuild { cmake { arguments += "-DCMAKE_BUILD_TYPE=RelWithDebInfo" } }
            }
        }
        release {
            if (starlingExperimental) {
                applicationIdSuffix = ".experimental"
            }
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
    implementation("androidx.core:core:1.19.1")
    // HTTP client for model downloads. Keep OkHttp and MockWebServer on the
    // same version so tests exercise one implementation.
    implementation("com.squareup.okhttp3:okhttp:5.5.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20260814")
    testImplementation("com.squareup.okhttp3:mockwebserver:5.5.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test:core:1.7.0")
    androidTestImplementation("androidx.test:rules:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
}
