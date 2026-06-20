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

/// Pipes ``Document`` values through chunking, embedding, and upsert to a
/// ``VectorStore``.
public protocol Indexer: Sendable {
    /// Indexes a batch of documents. Implementations should be safe to
    /// call repeatedly with the same documents; the resulting record ids
    /// must collide so that re-indexing replaces rather than duplicates.
    func index(_ documents: [Document]) async throws
}

extension Indexer {
    /// Convenience for indexing a single document.
    public func index(_ document: Document) async throws {
        try await index([document])
    }
}

/// The default ``Indexer`` — chunks each document, batch-embeds the chunk
/// texts, and upserts the resulting ``VectorRecord`` values into a
/// ``VectorStore``.
///
/// Each produced record's ``VectorRecord/metadata`` carries the chunk's
/// inherited metadata plus ``VectorMetadataKeys/documentID`` and
/// ``VectorMetadataKeys/chunkIndex``.
///
/// Embeddings come from the swift-embeddings `EmbeddingModel`; this library
/// does not implement embedding clients.
public struct DefaultIndexer: Indexer {
    public let chunker: any Chunker
    public let embeddingModel: any EmbeddingModel
    public let store: any VectorStore
    public let embedOptions: EmbeddingOptions
    public let idStrategy: @Sendable (Chunk) -> String

    /// Creates a default indexer.
    ///
    /// - Parameters:
    ///   - chunker: Splits each ``Document`` into chunks.
    ///   - embeddingModel: Embeds chunk texts. Called once per `index(_:)`
    ///     call with the full batch of chunks.
    ///   - store: Receives the produced records.
    ///   - embedOptions: Forwarded to every embed call.
    ///   - idStrategy: Computes the stable id for each record. Defaults
    ///     to `"<documentID>#<chunkIndex>"`, which guarantees re-indexing
    ///     the same document replaces existing records.
    public init(
        chunker: any Chunker,
        embeddingModel: any EmbeddingModel,
        store: any VectorStore,
        embedOptions: EmbeddingOptions = EmbeddingOptions(),
        idStrategy: @escaping @Sendable (Chunk) -> String = { "\($0.documentID)#\($0.index)" }
    ) {
        self.chunker = chunker
        self.embeddingModel = embeddingModel
        self.store = store
        self.embedOptions = embedOptions
        self.idStrategy = idStrategy
    }

    public func index(_ documents: [Document]) async throws {
        var chunks: [Chunk] = []
        for document in documents {
            let documentChunks = try await chunker.chunk(document)
            chunks.append(contentsOf: documentChunks)
        }
        guard chunks.isEmpty == false else { return }

        let texts = chunks.map { $0.text }
        let embeddings = try await embeddingModel.embed(.strings(texts), options: embedOptions)
        guard embeddings.vectors.count == chunks.count else {
            throw VectorStoreError.embeddingCountMismatch(
                expected: chunks.count,
                got: embeddings.vectors.count
            )
        }

        let records = zip(chunks, embeddings.vectors).map { chunk, vector -> VectorRecord in
            var metadata = chunk.metadata
            metadata[VectorMetadataKeys.documentID] = .string(chunk.documentID)
            metadata[VectorMetadataKeys.chunkIndex] = .integer(chunk.index)
            return VectorRecord(
                id: idStrategy(chunk),
                vector: vector,
                text: chunk.text,
                metadata: metadata
            )
        }
        try await store.upsert(records)
    }
}
