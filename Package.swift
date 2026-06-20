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
        ),
        .executable(
            name: "e5-embed-similarity",
            targets: ["E5EmbedSimilarityCLI"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0")
    ],
    targets: [
        .target(
            name: "E5EmbeddingCore",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "E5EmbedCLI",
            dependencies: ["E5EmbeddingCore"]
        ),
        .executableTarget(
            name: "E5EmbedSimilarityCLI",
            dependencies: ["E5EmbeddingCore"]
        ),
        .testTarget(
            name: "E5EmbeddingCoreTests",
            dependencies: ["E5EmbeddingCore"]
        )
    ]
)
