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

/// Shared rendering of retrieved matches into a numbered text block.
///
/// Both ``RetrievalTool`` and ``RetrievalContext`` ground a model with the
/// same passage layout so the model sees a consistent shape regardless of how
/// retrieval was triggered.
enum RetrievalFormatting {
    /// Renders `matches` as a numbered list, optionally prefixed by `header`.
    ///
    /// Returns an empty string when `matches` is empty so callers can decide
    /// whether to inject anything at all.
    static func passages(_ matches: [VectorMatch], header: String) -> String {
        guard matches.isEmpty == false else { return "" }
        var lines: [String] = []
        if header.isEmpty == false {
            lines.append(header)
        }
        for (offset, match) in matches.enumerated() {
            lines.append("[\(offset + 1)] \(match.record.text)")
        }
        return lines.joined(separator: "\n")
    }
}
