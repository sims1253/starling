// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "StarlingVoiceCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "StarlingVoiceCore", targets: ["StarlingVoiceCore"]),
    ],
    targets: [
        .target(
            name: "StarlingVoiceCore",
            path: "Sources/Core"
        ),
        .testTarget(
            name: "StarlingVoiceCoreTests",
            dependencies: ["StarlingVoiceCore"],
            path: "Tests/CoreTests"
        ),
    ]
)
