# ``FoundationModelsRetrieval``

Retrieval-augmented generation for Apple's Foundation Models — a vector store
and a `Tool` that grounds any `LanguageModelSession`.

## Overview

Apple's **FoundationModels** owns generation. This library is the missing piece:
it lets a `LanguageModelSession` retrieve from *your own* corpus. Index your
documents into a ``VectorStore``, then ground the model one of two ways — or
both, since they compose:

- **On demand** — add a ``RetrievalTool`` to a session's `tools:`. The model
  retrieves whenever it decides it needs context.
- **Upfront** — use ``RetrievalContext`` to fold retrieved passages directly
  into the session's `Instructions` or a `Prompt` before the model answers.

Embeddings and reranking come from the
[swift-embeddings](https://github.com/Dean151/swift-embeddings) package — this
library does not implement embedding clients.

### Indexing a corpus

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

### Grounding a session with the tool

```swift
import FoundationModels

let retriever = DefaultRetriever(embeddingModel: embeddingModel, store: store)
let tool = RetrievalTool(retriever: retriever)

let session = LanguageModelSession(model: .default, tools: [tool])
let answer = try await session.respond(to: "What is our refund policy?")

// Build a citations UI from what actually grounded the answer:
let citations = tool.lastMatches
```

### Grounding a prompt upfront

```swift
let context = RetrievalContext(retriever: retriever)
let answer = try await session.respondGrounded(
    to: "How many vacation days do I get?",
    using: context
)
```

## Topics

### Ingestion

- ``Document``
- ``Chunk``
- ``Chunker``
- ``RecursiveTextChunker``
- ``Indexer``
- ``DefaultIndexer``

### Storage

- ``VectorStore``
- ``InMemoryVectorStore``
- ``VectorRecord``
- ``VectorMatch``
- ``VectorMetadataKeys``
- ``VectorStoreError``

### Retrieval

- ``Retriever``
- ``RetrievalQuery``
- ``DefaultRetriever``
- ``RerankingRetriever``

### Foundation Models grounding

- ``RetrievalTool``
- ``RetrievalArguments``
- ``RetrievalContext``
