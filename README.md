# foundation-models-retrieval

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FDean151%2Ffoundation-models-retrieval%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/Dean151/foundation-models-retrieval)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FDean151%2Ffoundation-models-retrieval%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/Dean151/foundation-models-retrieval)

**Retrieval-augmented generation for Apple's Foundation Models** — a vector
store and a `Tool` that grounds any `LanguageModelSession`.

Apple's **FoundationModels** owns generation. This library is the missing
piece: it lets a `LanguageModelSession` retrieve from *your own* corpus by
adding a tool to `tools:`, or by folding retrieved passages into a prompt.

- **FoundationModels-native.** `RetrievalTool` conforms to FoundationModels'
  own `Tool` protocol with `@Generable` arguments — no bespoke agent
  abstraction to learn. Drop it into `tools:` and the model retrieves on demand.
- **Two ways to ground, and they compose.** A tool for on-demand lookups, and
  `RetrievalContext` for folding passages into `Instructions`/`Prompt` upfront.
- **Bring your own embeddings.** Embedding and reranking come from
  [swift-embeddings](https://github.com/Dean151/swift-embeddings); this library
  does not reimplement embedding clients.
- **Swift 6, fully Sendable.** Platforms floor: iOS 27 / macOS 27 / visionOS 27
  / watchOS 27 (the FoundationModels `Tool` API).

## Installation

```swift
.package(url: "https://github.com/Dean151/foundation-models-retrieval.git", from: "0.1.0"),
```

```swift
.target(name: "App", dependencies: [
    .product(name: "FoundationModelsRetrieval", package: "foundation-models-retrieval"),
])
```

## Indexing a corpus

Chunk your documents, embed them with a `swift-embeddings` model, and store the
vectors. `DefaultIndexer` wires the three steps together:

```swift
import Embeddings
import FoundationModelsRetrieval

let store = InMemoryVectorStore()
let embeddingModel = VoyageEmbeddingModel(model: .voyage3, apiKey: key)

let indexer = DefaultIndexer(
    chunker: RecursiveTextChunker(),
    embeddingModel: embeddingModel,
    store: store
)

try await indexer.index([
    Document(id: "handbook", text: handbookText),
    Document(id: "faq", text: faqText),
])
```

Re-indexing the same `Document` id replaces its records rather than duplicating
them, so indexing is safe to repeat.

## Grounding a session with the tool

Add a `RetrievalTool` to a session and the model decides when to retrieve:

```swift
import FoundationModels

let retriever = DefaultRetriever(embeddingModel: embeddingModel, store: store)
let tool = RetrievalTool(retriever: retriever)

let session = LanguageModelSession(model: .default, tools: [tool])
let answer = try await session.respond(to: "What is our refund policy?")
```

The tool returns a lean numbered passage block to the model. The raw matches —
with scores and metadata — are exposed separately so you can build a citations
UI without paying a structured-output token cost in the prompt:

```swift
let citations = tool.lastMatches          // pull after a response
// or stream them as they arrive:
let tool = RetrievalTool(retriever: retriever) { matches in
    render(citations: matches)
}
```

## Grounding a prompt upfront

When you'd rather guarantee context *before* the model answers, use
`RetrievalContext` to fold retrieved passages into a `Prompt` or the session
`Instructions`:

```swift
let context = RetrievalContext(retriever: retriever)

// One-call convenience: retrieve, prepend passages, respond.
let answer = try await session.respondGrounded(
    to: "How many vacation days do I get?",
    using: context
)

// Or build the pieces yourself:
let prompt = try await context.prompt(augmenting: "How many vacation days do I get?")
let instructions = try await context.instructions(for: "HR policy") {
    Instructions("You are a helpful HR assistant.")
}
```

The two approaches compose: ground system `Instructions` once and still expose
a `RetrievalTool` for follow-up lookups.

## Reranking

Embedding similarity is cheap but coarse. Wrap any `Retriever` in a
`RerankingRetriever` to fetch a wider candidate pool and reorder it with a
`swift-embeddings` `RerankModel`:

```swift
let reranking = RerankingRetriever(
    base: DefaultRetriever(embeddingModel: embeddingModel, store: store),
    rerankModel: CohereRerankModel(model: .rerankV35, apiKey: key)
)
let tool = RetrievalTool(retriever: reranking)
```

## Custom storage

`InMemoryVectorStore` is a volatile cosine-similarity store, ideal for tests
and small corpora. For persistence or an external vector database, conform your
own type to `VectorStore` — the rest of the pipeline is unchanged.

## Documentation

Browse the API on the
[Swift Package Index](https://swiftpackageindex.com/Dean151/foundation-models-retrieval/main/documentation/foundationmodelsretrieval)
or build it locally with `swift package generate-documentation`.

## License

MIT. See [LICENSE](LICENSE).
