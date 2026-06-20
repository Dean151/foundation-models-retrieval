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

import Synchronization

/// A storage backend for embedded ``VectorRecord`` values.
///
/// Implementations are agnostic of how vectors are produced. Callers pass
/// in already-embedded records; the store returns top-K matches with
/// their similarity score. Conforming stores must preserve
/// ``VectorRecord/metadata`` round-trip verbatim.
public protocol VectorStore: Sendable {
    /// Inserts or replaces a batch of records keyed by ``VectorRecord/id``.
    func upsert(_ records: [VectorRecord]) async throws

    /// Returns up to `topK` records ranked by descending similarity to
    /// `query`. Implementations decide their own similarity metric.
    func search(query: [Double], topK: Int) async throws -> [VectorMatch]

    /// Removes records with the supplied ids. Ids that do not exist are
    /// silently ignored.
    func delete(ids: [String]) async throws

    /// Removes every entry from the store.
    func clear() async throws
}

extension VectorStore {
    /// Convenience for upserting a single record.
    public func upsert(_ record: VectorRecord) async throws {
        try await upsert([record])
    }
}

/// A volatile cosine-similarity ``VectorStore`` backed by in-process arrays.
public final class InMemoryVectorStore: VectorStore, Sendable {
    private struct State {
        var entries: [String: VectorRecord] = [:]
        var order: [String] = []
    }

    private let state: Mutex<State>

    /// Creates an empty in-memory vector store.
    public init() {
        self.state = Mutex(State())
    }

    public func upsert(_ records: [VectorRecord]) async throws {
        state.withLock { state in
            for record in records {
                if state.entries[record.id] == nil {
                    state.order.append(record.id)
                }
                state.entries[record.id] = record
            }
        }
    }

    public func search(query: [Double], topK: Int) async throws -> [VectorMatch] {
        guard topK > 0 else { return [] }
        let snapshot = state.withLock { state -> [VectorRecord] in
            state.order.compactMap { state.entries[$0] }
        }
        let scored = snapshot.map { record -> VectorMatch in
            let score = Self.cosineSimilarity(query, record.vector)
            return VectorMatch(record: record, score: score)
        }
        return scored.sorted { $0.score > $1.score }.prefix(topK).map { $0 }
    }

    public func delete(ids: [String]) async throws {
        state.withLock { state in
            let toRemove = Set(ids)
            for id in toRemove {
                state.entries.removeValue(forKey: id)
            }
            state.order.removeAll { toRemove.contains($0) }
        }
    }

    public func clear() async throws {
        state.withLock { state in
            state.entries.removeAll()
            state.order.removeAll()
        }
    }

    /// Cosine similarity in [-1, 1]; returns 0 when either vector is empty
    /// or has zero magnitude.
    private static func cosineSimilarity(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.isEmpty == false, rhs.isEmpty == false else { return 0 }
        var dot = 0.0
        var lhsNorm = 0.0
        var rhsNorm = 0.0
        for (l, r) in zip(lhs, rhs) {
            dot += l * r
            lhsNorm += l * l
            rhsNorm += r * r
        }
        let denom = lhsNorm.squareRoot() * rhsNorm.squareRoot()
        return denom > 0 ? dot / denom : 0
    }
}
