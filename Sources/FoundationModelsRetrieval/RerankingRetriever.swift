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

/// A ``Retriever`` that refines another retriever's results with a
/// swift-embeddings `RerankModel`.
///
/// Embedding similarity is cheap but coarse. `RerankingRetriever` fetches a
/// wider candidate pool from a `base` retriever — `topK × candidateMultiplier`
/// matches — then asks a `RerankModel` to score each candidate's text against
/// the query, and returns the best `topK` by relevance. The returned
/// ``VectorMatch/score`` values are the rerank relevance scores, not the base
/// similarity scores.
public struct RerankingRetriever: Retriever {
    /// The first-stage retriever that produces the candidate pool.
    public let base: any Retriever
    /// The model used to reorder the candidates.
    public let rerankModel: any RerankModel
    /// How many times `topK` candidates to fetch from `base` before reranking.
    /// Must be >= 1.
    public let candidateMultiplier: Int
    /// Rerank options forwarded to the model. Its `topN` is set by the
    /// retriever from the query's `topK`, overriding any value here.
    public let rerankOptions: RerankOptions

    /// Creates a reranking retriever.
    ///
    /// - Parameters:
    ///   - base: The first-stage retriever to refine.
    ///   - rerankModel: The model used to reorder candidates.
    ///   - candidateMultiplier: How many times `topK` candidates to pull from
    ///     `base` before reranking. Defaults to 4.
    ///   - rerankOptions: Extra rerank options. Defaults to empty.
    public init(
        base: any Retriever,
        rerankModel: any RerankModel,
        candidateMultiplier: Int = 4,
        rerankOptions: RerankOptions = RerankOptions()
    ) {
        self.base = base
        self.rerankModel = rerankModel
        self.candidateMultiplier = max(1, candidateMultiplier)
        self.rerankOptions = rerankOptions
    }

    public func retrieve(_ query: RetrievalQuery) async throws -> [VectorMatch] {
        guard query.topK > 0, query.text.isEmpty == false else { return [] }

        var candidateQuery = query
        candidateQuery.topK = query.topK * candidateMultiplier
        let candidates = try await base.retrieve(candidateQuery)
        guard candidates.isEmpty == false else { return [] }

        var options = rerankOptions
        options.topN = query.topK
        let ranking = try await rerankModel.rerank(
            query.text,
            documents: candidates.map(\.record.text),
            options: options
        )

        var reranked: [VectorMatch] = []
        reranked.reserveCapacity(min(query.topK, ranking.results.count))
        for ranked in ranking.results {
            guard candidates.indices.contains(ranked.index) else { continue }
            reranked.append(VectorMatch(record: candidates[ranked.index].record, score: ranked.relevanceScore))
            if reranked.count == query.topK { break }
        }
        return reranked
    }
}
