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

import Embeddings
@testable import FoundationModelsRetrieval

/// An embedding model that returns a preconfigured vector per text, falling
/// back to a zero vector for unknown inputs.
struct StubEmbeddingModel: EmbeddingModel {
    var vectors: [String: [Double]]
    var dimension: Int

    init(vectors: [String: [Double]], dimension: Int) {
        self.vectors = vectors
        self.dimension = dimension
    }

    func embed(_ input: EmbeddingInput, options: EmbeddingOptions) async throws -> Embeddings {
        let produced = input.values.map { vectors[$0] ?? Array(repeating: 0.0, count: dimension) }
        return Embeddings(model: "stub", vectors: produced)
    }
}

/// An embedding model that returns a fixed number of vectors regardless of
/// input count — used to exercise count-mismatch handling.
struct MiscountingEmbeddingModel: EmbeddingModel {
    var returnedVectorCount: Int

    func embed(_ input: EmbeddingInput, options: EmbeddingOptions) async throws -> Embeddings {
        let produced = (0..<returnedVectorCount).map { _ in [1.0, 0.0] }
        return Embeddings(model: "stub", vectors: produced)
    }
}

/// A retriever that returns a fixed list of matches, honoring `topK` and
/// `minScore` from the query.
struct StubRetriever: Retriever {
    var matches: [VectorMatch]

    func retrieve(_ query: RetrievalQuery) async throws -> [VectorMatch] {
        Array(matches.filter { $0.score >= query.minScore }.prefix(query.topK))
    }
}

/// A rerank model that reorders candidates by a supplied index order.
struct StubRerankModel: RerankModel {
    /// Candidate indices, most relevant first.
    var order: [Int]

    func rerank(_ query: String, documents: [String], options: RerankOptions) async throws -> RerankResults {
        let ranked = order.enumerated().map { rank, index in
            RankedDocument(index: index, relevanceScore: Double(order.count - rank))
        }
        return RerankResults(model: "stub", results: ranked)
    }
}

extension VectorMatch {
    /// Convenience for building a match from raw text and score in tests.
    static func make(id: String, text: String, score: Double) -> VectorMatch {
        VectorMatch(record: VectorRecord(id: id, vector: [], text: text), score: score)
    }
}
