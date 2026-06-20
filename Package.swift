// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "foundation-models-retrieval",
    platforms: [
        .macOS("27.0"),
        .iOS("27.0"),
        .visionOS("27.0"),
        .watchOS("27.0"),
    ],
    products: [
        // Retrieval-augmented generation for Apple's Foundation Models: a vector
        // store and a `Tool` that grounds any `LanguageModelSession`.
        .library(
            name: "FoundationModelsRetrieval",
            targets: ["FoundationModelsRetrieval"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/Dean151/swift-embeddings", from: "0.1.0"),
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        .target(
            name: "FoundationModelsRetrieval",
            dependencies: [
                .product(name: "Embeddings", package: "swift-embeddings"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .testTarget(
            name: "FoundationModelsRetrievalTests",
            dependencies: ["FoundationModelsRetrieval"],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
    ],
    swiftLanguageModes: [.v6]
)
