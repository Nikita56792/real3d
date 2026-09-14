// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Real3D",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "Real3D", targets: ["Real3D"]),
        .executable(name: "real3d-bench", targets: ["real3d-bench"]),
    ],
    targets: [
        // Core pipeline: screen capture, depth inference, Metal stereo synthesis.
        .target(
            name: "Real3DCore",
            path: "Sources/Real3DCore",
            exclude: ["Shaders.metal"],
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        ),
        // The GUI application.
        .executableTarget(
            name: "Real3D",
            dependencies: ["Real3DCore"],
            path: "Sources/Real3D"
        ),
        // CLI: inspect a Core ML depth model and benchmark it on ANE / GPU / CPU.
        .executableTarget(
            name: "real3d-bench",
            dependencies: ["Real3DCore"],
            path: "Sources/real3d-bench"
        ),
    ]
)
