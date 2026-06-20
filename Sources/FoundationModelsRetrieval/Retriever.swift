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

/// A query to a ``Retriever``.
public struct RetrievalQuery: Sendable {
    /// The text to embed and search for.
    public var text: String
    /// Maximum number of matches to return. Must be > 0.
    public var topK: Int
    /// Drops matches whose score is strictly less than this value.
    /// Stores using cosine similarity should pass a value in `[-1, 1]`.
    public var minScore: Double

    /// Creates a retrieval query.
    public init(text: String, topK: Int = 5, minScore: Double = 0.0) {
        self.text = text
        self.topK = topK
        self.minScore = minScore
    }
}

/// Looks up relevant records from a ``VectorStore``.
public protocol Retriever: Sendable {
    /// Returns up to ``RetrievalQuery/topK`` matches for the supplied
    /// query, filtered by ``RetrievalQuery/minScore``.
    func retrieve(_ query: RetrievalQuery) async throws -> [VectorMatch]
}

/// The default ``Retriever`` — embeds the query text once and forwards
/// to a ``VectorStore``'s search method, then filters by `minScore`.
public struct DefaultRetriever: Retriever {
    public let embeddingModel: any EmbeddingModel
    public let store: any VectorStore
    public let embedOptions: EmbeddingOptions

    /// Creates a default retriever.
    public init(
        embeddingModel: any EmbeddingModel,
        store: any VectorStore,
        embedOptions: EmbeddingOptions = EmbeddingOptions()
    ) {
        self.embeddingModel = embeddingModel
        self.store = store
        self.embedOptions = embedOptions
    }

    public func retrieve(_ query: RetrievalQuery) async throws -> [VectorMatch] {
        guard query.topK > 0, query.text.isEmpty == false else { return [] }
        let embeddings = try await embeddingModel.embed(.string(query.text), options: embedOptions)
        guard let vector = embeddings.vectors.first else {
            throw VectorStoreError.missingEmbedding
        }
        let matches = try await store.search(query: vector, topK: query.topK)
        return matches.filter { $0.score >= query.minScore }
    }
}
