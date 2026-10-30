// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "tlc-mlx-runner",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.32.3"),
    ],
    targets: [
        .executableTarget(
            name: "tlc-mlx-runner",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ]
        ),
    ]
)
