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

#if canImport(FoundationEssentials)
import FoundationEssentials
#else
import Foundation
#endif

/// Splits a ``Document`` into ``Chunk`` values ready to be embedded.
///
/// Implementations decide their own splitting strategy (character count,
/// token count, sentence boundaries, semantic boundaries, etc.). Each
/// produced chunk should carry a 0-based contiguous ``Chunk/index`` and
/// preserve the document's metadata.
public protocol Chunker: Sendable {
    /// Splits a document into chunks. Returning an empty array is valid
    /// (e.g., for an empty document).
    func chunk(_ document: Document) async throws -> [Chunk]
}

/// A character-budget splitter that walks a hierarchy of separators
/// (paragraph → line → sentence → word) before falling back to a hard
/// character cut.
///
/// `RecursiveTextChunker` aims to keep semantically meaningful pieces
/// together: it tries the most coarse separator first and only descends
/// to a finer one when a piece is still over budget. Successive chunks
/// overlap by ``overlapCharacters`` characters, which helps retrieval
/// recall near chunk boundaries.
public struct RecursiveTextChunker: Chunker {
    /// The soft upper bound on chunk length, measured in characters.
    public let maxCharacters: Int
    /// The number of trailing characters from one chunk that are repeated
    /// at the start of the next.
    public let overlapCharacters: Int

    /// Creates a recursive text chunker.
    ///
    /// - Parameters:
    ///   - maxCharacters: Soft upper bound on chunk length. Must be > 0.
    ///   - overlapCharacters: Characters of overlap between adjacent
    ///     chunks. Clamped to `[0, maxCharacters - 1]`.
    public init(maxCharacters: Int = 1000, overlapCharacters: Int = 100) {
        precondition(maxCharacters > 0, "maxCharacters must be > 0")
        self.maxCharacters = maxCharacters
        self.overlapCharacters = max(0, min(overlapCharacters, maxCharacters - 1))
    }

    public func chunk(_ document: Document) async throws -> [Chunk] {
        let text = document.text
        guard text.isEmpty == false else { return [] }

        let pieces = Self.split(text, separators: ["\n\n", "\n", ". ", " "], limit: maxCharacters)
        let merged = mergeWithOverlap(pieces)
        return merged.enumerated().map { offset, body in
            Chunk(
                documentID: document.id,
                index: offset,
                text: body,
                metadata: document.metadata
            )
        }
    }

    /// Greedily packs `pieces` into chunks of at most ``maxCharacters``,
    /// prefixing each non-first chunk with ``overlapCharacters`` trailing
    /// characters of the previous chunk.
    private func mergeWithOverlap(_ pieces: [String]) -> [String] {
        var chunks: [String] = []
        var current = ""
        for piece in pieces {
            if current.isEmpty {
                current = piece
                continue
            }
            if current.count + piece.count + 1 <= maxCharacters {
                current += " " + piece
            } else {
                chunks.append(current)
                let overlap = Self.tail(of: current, characters: overlapCharacters)
                current = overlap.isEmpty ? piece : overlap + " " + piece
            }
        }
        if current.isEmpty == false {
            chunks.append(current)
        }
        return chunks
    }

    /// Returns the trailing `characters` characters of `string`. Returns
    /// an empty string when `characters <= 0`, and the whole string when
    /// it is shorter than the requested suffix.
    private static func tail(of string: String, characters: Int) -> String {
        guard characters > 0 else { return "" }
        guard string.count > characters else { return string }
        return String(string.suffix(characters))
    }

    /// Recursively splits `text` by trying each separator in order and
    /// descending into pieces still longer than `limit`. The final
    /// fallback is a hard character cut.
    private static func split(_ text: String, separators: [String], limit: Int) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty == false else { return [] }
        if trimmed.count <= limit {
            return [trimmed]
        }
        guard separators.isEmpty == false else {
            return hardCut(trimmed, limit: limit)
        }
        let separator = separators[0]
        let remaining = Array(separators.dropFirst())
        let parts = trimmed.components(separatedBy: separator)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.isEmpty == false }
        guard parts.count > 1 else {
            return split(trimmed, separators: remaining, limit: limit)
        }
        var output: [String] = []
        for part in parts {
            if part.count <= limit {
                output.append(part)
            } else {
                output.append(contentsOf: split(part, separators: remaining, limit: limit))
            }
        }
        return output
    }

    /// Slices `text` into pieces of at most `limit` characters with no
    /// regard for word boundaries — used as the final fallback when no
    /// separator splits it small enough.
    private static func hardCut(_ text: String, limit: Int) -> [String] {
        var output: [String] = []
        var index = text.startIndex
        while index < text.endIndex {
            let next = text.index(index, offsetBy: limit, limitedBy: text.endIndex) ?? text.endIndex
            output.append(String(text[index..<next]))
            index = next
        }
        return output
    }
}
