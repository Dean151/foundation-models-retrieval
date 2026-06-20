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

/// A pre-chunk input to an ``Indexer``.
///
/// A document represents one "thing the caller knows about" — a page, an
/// email, a wiki entry, a transcript turn. The ``Indexer`` is responsible
/// for splitting it into ``Chunk`` values via a ``Chunker``.
public struct Document: Sendable {
    /// A stable identifier. The ``Indexer`` stamps this onto every chunk
    /// produced from the document under ``VectorMetadataKeys/documentID``.
    public let id: String
    /// The text body of the document.
    public let text: String
    /// Optional metadata carried over to every chunk produced from this
    /// document.
    public let metadata: [String: JSONValue]

    /// Creates a document.
    public init(id: String, text: String, metadata: [String: JSONValue] = [:]) {
        self.id = id
        self.text = text
        self.metadata = metadata
    }
}
