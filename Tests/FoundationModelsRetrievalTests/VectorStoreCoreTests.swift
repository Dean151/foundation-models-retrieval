//
//  MIT License
//
//  Copyright (c) 2026 Thomas Durand
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
//

import Testing
import Embeddings
@testable import FoundationModelsRetrieval

@Suite struct InMemoryVectorStoreTests {
    @Test func ranksByCosineSimilarityDescending() async throws {
        let store = InMemoryVectorStore()
        try await store.upsert([
            VectorRecord(id: "a", vector: [1, 0], text: "a"),
            VectorRecord(id: "b", vector: [0, 1], text: "b"),
            VectorRecord(id: "c", vector: [0.9, 0.1], text: "c"),
        ])
        let matches = try await store.search(query: [1, 0], topK: 3)
        #expect(matches.map(\.record.id) == ["a", "c", "b"])
        #expect(matches[0].score > matches[1].score)
    }

    @Test func truncatesToTopK() async throws {
        let store = InMemoryVectorStore()
        try await store.upsert((0..<5).map { VectorRecord(id: "\($0)", vector: [Double($0), 1], text: "\($0)") })
        let matches = try await store.search(query: [1, 1], topK: 2)
        #expect(matches.count == 2)
    }

    @Test func upsertReplacesAndPreservesMetadata() async throws {
        let store = InMemoryVectorStore()
        try await store.upsert(VectorRecord(id: "x", vector: [1, 0], text: "old"))
        try await store.upsert(VectorRecord(id: "x", vector: [1, 0], text: "new", metadata: ["k": .string("v")]))
        let matches = try await store.search(query: [1, 0], topK: 5)
        #expect(matches.count == 1)
        #expect(matches[0].record.text == "new")
        #expect(matches[0].record.metadata["k"] == .string("v"))
    }

    @Test func deleteAndClear() async throws {
        let store = InMemoryVectorStore()
        try await store.upsert([
            VectorRecord(id: "a", vector: [1, 0], text: "a"),
            VectorRecord(id: "b", vector: [0, 1], text: "b"),
        ])
        try await store.delete(ids: ["a", "missing"])
        #expect(try await store.search(query: [1, 0], topK: 5).map(\.record.id) == ["b"])
        try await store.clear()
        #expect(try await store.search(query: [1, 0], topK: 5).isEmpty)
    }

    @Test func emptyQueryVectorScoresZero() async throws {
        let store = InMemoryVectorStore()
        try await store.upsert(VectorRecord(id: "a", vector: [1, 0], text: "a"))
        let matches = try await store.search(query: [], topK: 1)
        #expect(matches.first?.score == 0)
    }
}

@Suite struct RecursiveTextChunkerTests {
    @Test func emptyDocumentProducesNoChunks() async throws {
        let chunks = try await RecursiveTextChunker().chunk(Document(id: "d", text: ""))
        #expect(chunks.isEmpty)
    }

    @Test func shortDocumentIsASingleChunk() async throws {
        let chunks = try await RecursiveTextChunker(maxCharacters: 100).chunk(Document(id: "d", text: "hello world"))
        #expect(chunks.count == 1)
        #expect(chunks[0].index == 0)
        #expect(chunks[0].documentID == "d")
        #expect(chunks[0].text == "hello world")
    }

    @Test func longDocumentSplitsIntoContiguousChunks() async throws {
        let paragraph = String(repeating: "word ", count: 200)
        let text = paragraph + "\n\n" + paragraph
        let chunks = try await RecursiveTextChunker(maxCharacters: 200, overlapCharacters: 20).chunk(
            Document(id: "d", text: text)
        )
        #expect(chunks.count > 1)
        #expect(chunks.map(\.index) == Array(0..<chunks.count))
        #expect(chunks.allSatisfy { $0.documentID == "d" })
    }

    @Test func inheritsDocumentMetadata() async throws {
        let chunks = try await RecursiveTextChunker().chunk(
            Document(id: "d", text: "body", metadata: ["src": .string("wiki")])
        )
        #expect(chunks[0].metadata["src"] == .string("wiki"))
    }
}

@Suite struct DefaultIndexerTests {
    @Test func indexesChunksWithProvenanceMetadata() async throws {
        let store = InMemoryVectorStore()
        let embedder = StubEmbeddingModel(vectors: ["hello world": [1, 0]], dimension: 2)
        let indexer = DefaultIndexer(
            chunker: RecursiveTextChunker(maxCharacters: 100),
            embeddingModel: embedder,
            store: store
        )
        try await indexer.index(Document(id: "doc1", text: "hello world"))

        let matches = try await store.search(query: [1, 0], topK: 5)
        #expect(matches.count == 1)
        let record = matches[0].record
        #expect(record.id == "doc1#0")
        #expect(record.metadata[VectorMetadataKeys.documentID] == .string("doc1"))
        #expect(record.metadata[VectorMetadataKeys.chunkIndex] == .integer(0))
    }

    @Test func reindexingReplacesRatherThanDuplicates() async throws {
        let store = InMemoryVectorStore()
        let embedder = StubEmbeddingModel(vectors: ["text": [1, 0]], dimension: 2)
        let indexer = DefaultIndexer(chunker: RecursiveTextChunker(), embeddingModel: embedder, store: store)
        try await indexer.index(Document(id: "d", text: "text"))
        try await indexer.index(Document(id: "d", text: "text"))
        #expect(try await store.search(query: [1, 0], topK: 5).count == 1)
    }

    @Test func throwsOnEmbeddingCountMismatch() async throws {
        let store = InMemoryVectorStore()
        let indexer = DefaultIndexer(
            chunker: RecursiveTextChunker(),
            embeddingModel: MiscountingEmbeddingModel(returnedVectorCount: 5),
            store: store
        )
        await #expect(throws: VectorStoreError.self) {
            try await indexer.index(Document(id: "d", text: "just one chunk"))
        }
    }

    @Test func emptyDocumentIndexesNothing() async throws {
        let store = InMemoryVectorStore()
        let indexer = DefaultIndexer(
            chunker: RecursiveTextChunker(),
            embeddingModel: StubEmbeddingModel(vectors: [:], dimension: 2),
            store: store
        )
        try await indexer.index(Document(id: "d", text: ""))
        #expect(try await store.search(query: [1, 0], topK: 5).isEmpty)
    }
}
