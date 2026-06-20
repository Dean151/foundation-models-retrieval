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

/// The canonical stored unit in a ``VectorStore``.
///
/// A record pairs an embedding `vector` with the `text` it was computed from
/// and an open-ended `metadata` bag the caller can use to attach provenance
/// or domain-specific information.
public struct VectorRecord: Sendable {
    /// A stable identifier; upserting with an existing `id` replaces the entry.
    public let id: String
    /// The embedding vector. Stores treat this opaquely; dimensionality is
    /// up to the embedder.
    public let vector: [Double]
    /// The text the vector was computed from. Returned verbatim by stores.
    public let text: String
    /// An open metadata bag, preserved round-trip by conforming stores.
    public let metadata: [String: JSONValue]

    /// Creates a vector record.
    public init(
        id: String,
        vector: [Double],
        text: String,
        metadata: [String: JSONValue] = [:]
    ) {
        self.id = id
        self.vector = vector
        self.text = text
        self.metadata = metadata
    }
}

/// Well-known keys the library writes into ``VectorRecord/metadata`` so
/// retrievers and callers can recover provenance.
///
/// These are conventions, not a schema — callers are free to add their own
/// keys. Conforming stores must preserve every key verbatim.
public enum VectorMetadataKeys {
    /// The originating ``Document/id`` for a record produced by an ``Indexer``.
    public static let documentID = "documentID"
    /// The 0-based index of the chunk within its source document.
    public static let chunkIndex = "chunkIndex"
    /// An optional URL identifying the source of the record.
    public static let sourceURL = "sourceURL"
}
