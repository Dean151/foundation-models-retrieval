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

/// A post-chunk, pre-embed slice of a ``Document``.
///
/// Chunks are produced by a ``Chunker`` and consumed by an ``Indexer``,
/// which embeds each chunk's `text` and writes a corresponding
/// ``VectorRecord`` to a ``VectorStore``.
public struct Chunk: Sendable {
    /// The originating document identifier.
    public let documentID: String
    /// The 0-based position of this chunk within its source document.
    public let index: Int
    /// The chunk text — what gets embedded.
    public let text: String
    /// Metadata inherited from the document plus any per-chunk additions.
    public let metadata: [String: JSONValue]

    /// Creates a chunk.
    public init(
        documentID: String,
        index: Int,
        text: String,
        metadata: [String: JSONValue] = [:]
    ) {
        self.documentID = documentID
        self.index = index
        self.text = text
        self.metadata = metadata
    }
}
