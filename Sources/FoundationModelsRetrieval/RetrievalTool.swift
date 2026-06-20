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

import FoundationModels
import Synchronization

/// The arguments a `LanguageModelSession` generates when it decides to call a
/// ``RetrievalTool``.
///
/// `@Generable` lets Foundation Models synthesize the tool's parameter schema
/// directly from this type — no hand-written JSON schema required.
@Generable
public struct RetrievalArguments: Sendable {
    /// The natural-language query the model wants to look up in the corpus.
    @Guide(description: "A natural-language query describing the information to look up in the knowledge base.")
    public var query: String

    /// An optional override for how many passages to return. When omitted, the
    /// tool uses its configured default.
    @Guide(description: "Optional maximum number of passages to retrieve. Leave empty to use the default.")
    public var topK: Int?

    /// Memberwise initializer (the `@Generable` macro provides the
    /// `GeneratedContent` initializer separately).
    public init(query: String, topK: Int? = nil) {
        self.query = query
        self.topK = topK
    }
}

/// A Foundation Models `Tool` that grounds a `LanguageModelSession` in your own
/// corpus.
///
/// Add it to a session's `tools:` and the model can retrieve relevant passages
/// on demand:
///
/// ```swift
/// let tool = RetrievalTool(retriever: retriever)
/// let session = LanguageModelSession(model: .default, tools: [tool])
/// let answer = try await session.respond(to: "What is our refund policy?")
/// // Inspect what grounded the answer:
/// let citations = tool.lastMatches
/// ```
///
/// The tool returns a numbered text block (lean, and well understood by the
/// model). The raw ``VectorMatch`` values — with scores and metadata — are
/// exposed separately via ``lastMatches`` and the `onResults` callback so your
/// app can build a citations UI without paying a structured-output token cost
/// in the prompt.
public struct RetrievalTool: Tool {
    public typealias Arguments = RetrievalArguments
    public typealias Output = String

    /// The tool name advertised to the model.
    public let name: String
    /// The natural-language description advertised to the model.
    public let description: String
    /// The retriever backing this tool.
    public let retriever: any Retriever
    /// The number of passages to retrieve when the model does not specify one.
    public let defaultTopK: Int
    /// Drops matches scoring below this threshold before returning them.
    public let minScore: Double
    /// The header prepended to the rendered passage block.
    public let header: String

    private let onResults: (@Sendable ([VectorMatch]) -> Void)?
    private let latest = MatchesBox()

    /// Creates a retrieval tool.
    ///
    /// - Parameters:
    ///   - retriever: The retriever queried on each tool call.
    ///   - name: The tool name advertised to the model.
    ///   - description: The tool description advertised to the model.
    ///   - defaultTopK: Passages to return when the model omits `topK`.
    ///   - minScore: Minimum score a match must meet to be returned.
    ///   - header: Header prepended to the rendered passage block.
    ///   - onResults: Optional callback invoked with the raw matches each time
    ///     the tool runs — handy for streaming citations as they arrive.
    public init(
        retriever: any Retriever,
        name: String = "searchKnowledge",
        description: String = "Search the knowledge base for passages relevant to a query.",
        defaultTopK: Int = 5,
        minScore: Double = 0.0,
        header: String = "Relevant passages:",
        onResults: (@Sendable ([VectorMatch]) -> Void)? = nil
    ) {
        self.retriever = retriever
        self.name = name
        self.description = description
        self.defaultTopK = defaultTopK
        self.minScore = minScore
        self.header = header
        self.onResults = onResults
    }

    /// The matches produced by the most recent ``call(arguments:)``.
    ///
    /// Empty until the tool has run at least once. The value is shared across
    /// copies of the tool, so reading it after a session response reflects the
    /// model's last retrieval.
    public var lastMatches: [VectorMatch] {
        latest.value
    }

    public func call(arguments: RetrievalArguments) async throws -> String {
        let topK = arguments.topK.map { max(1, $0) } ?? defaultTopK
        let query = RetrievalQuery(text: arguments.query, topK: topK, minScore: minScore)
        let matches = try await retriever.retrieve(query)

        latest.value = matches
        onResults?(matches)

        let block = RetrievalFormatting.passages(matches, header: header)
        return block.isEmpty ? "No relevant passages were found." : block
    }
}

/// A `Sendable` reference cell for the tool's most recent matches.
///
/// `RetrievalTool` is a value type, but Foundation Models copies tools when it
/// stores them. Sharing the latest matches through a reference cell means
/// ``RetrievalTool/lastMatches`` reflects the model's retrieval regardless of
/// which copy ran.
private final class MatchesBox: Sendable {
    private let storage = Mutex<[VectorMatch]>([])

    var value: [VectorMatch] {
        get { storage.withLock { $0 } }
        set { storage.withLock { $0 = newValue } }
    }
}
