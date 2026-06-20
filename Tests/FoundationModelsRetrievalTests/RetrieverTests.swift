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

@Suite struct DefaultRetrieverTests {
    private func makeRetriever() async throws -> DefaultRetriever {
        let store = InMemoryVectorStore()
        try await store.upsert([
            VectorRecord(id: "a", vector: [1, 0], text: "apples"),
            VectorRecord(id: "b", vector: [0, 1], text: "bananas"),
        ])
        let embedder = StubEmbeddingModel(vectors: ["fruit": [1, 0.05]], dimension: 2)
        return DefaultRetriever(embeddingModel: embedder, store: store)
    }

    @Test func retrievesOrderedMatches() async throws {
        let retriever = try await makeRetriever()
        let matches = try await retriever.retrieve(RetrievalQuery(text: "fruit", topK: 2))
        #expect(matches.first?.record.id == "a")
    }

    @Test func filtersByMinScore() async throws {
        let retriever = try await makeRetriever()
        let matches = try await retriever.retrieve(RetrievalQuery(text: "fruit", topK: 5, minScore: 0.5))
        #expect(matches.allSatisfy { $0.score >= 0.5 })
        #expect(matches.map(\.record.id) == ["a"])
    }

    @Test func emptyQueryReturnsNothing() async throws {
        let retriever = try await makeRetriever()
        #expect(try await retriever.retrieve(RetrievalQuery(text: "", topK: 5)).isEmpty)
    }

    @Test func nonPositiveTopKReturnsNothing() async throws {
        let retriever = try await makeRetriever()
        #expect(try await retriever.retrieve(RetrievalQuery(text: "fruit", topK: 0)).isEmpty)
    }
}

@Suite struct RerankingRetrieverTests {
    @Test func reordersByRerankScoreAndTruncates() async throws {
        let base = StubRetriever(matches: [
            .make(id: "a", text: "a", score: 0.9),
            .make(id: "b", text: "b", score: 0.8),
            .make(id: "c", text: "c", score: 0.7),
        ])
        // Rerank prefers candidate index 2 (c), then 0 (a), then 1 (b).
        let reranker = RerankingRetriever(
            base: base,
            rerankModel: StubRerankModel(order: [2, 0, 1]),
            candidateMultiplier: 4
        )
        let matches = try await reranker.retrieve(RetrievalQuery(text: "q", topK: 2))
        #expect(matches.map(\.record.id) == ["c", "a"])
        // Scores are the rerank relevance scores, descending.
        #expect(matches[0].score > matches[1].score)
    }

    @Test func emptyCandidatePoolReturnsNothing() async throws {
        let reranker = RerankingRetriever(
            base: StubRetriever(matches: []),
            rerankModel: StubRerankModel(order: [])
        )
        #expect(try await reranker.retrieve(RetrievalQuery(text: "q", topK: 3)).isEmpty)
    }
}
