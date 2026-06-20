// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "e5-coreml-swift-embedding-cli",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "E5EmbeddingCore",
            targets: ["E5EmbeddingCore"]
        ),
        .executable(
            name: "e5-embed",
            targets: ["E5EmbedCLI"]
        )
    ],
    targets: [
        .target(
            name: "E5EmbeddingCore"
        ),
        .executableTarget(
            name: "E5EmbedCLI",
            dependencies: ["E5EmbeddingCore"]
        ),
        .testTarget(
            name: "E5EmbeddingCoreTests",
            dependencies: ["E5EmbeddingCore"]
        )
    ]
)
