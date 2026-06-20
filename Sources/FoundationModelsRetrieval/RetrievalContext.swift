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

/// Folds retrieved passages directly into a session's `Instructions` or a
/// `Prompt` — upfront grounding, as opposed to the on-demand grounding of
/// ``RetrievalTool``.
///
/// Use this when you want to guarantee the model sees relevant context *before*
/// it answers, rather than relying on it to call a tool. The two compose: many
/// apps ground system `Instructions` once and also expose a ``RetrievalTool``
/// for follow-up lookups.
///
/// ```swift
/// let context = RetrievalContext(retriever: retriever)
/// let prompt = try await context.prompt(augmenting: "How many vacation days do I get?")
/// let answer = try await session.respond(to: prompt)
/// ```
public struct RetrievalContext: Sendable {
    /// The retriever queried to gather grounding passages.
    public let retriever: any Retriever
    /// The number of passages to retrieve.
    public let topK: Int
    /// Drops matches scoring below this threshold.
    public let minScore: Double
    /// The header prepended to the rendered passage block.
    public let header: String

    /// Creates a retrieval context helper.
    ///
    /// - Parameters:
    ///   - retriever: The retriever queried for grounding passages.
    ///   - topK: The number of passages to retrieve. Defaults to 3.
    ///   - minScore: Minimum score a match must meet to be included.
    ///   - header: Header prepended to the rendered passage block.
    public init(
        retriever: any Retriever,
        topK: Int = 3,
        minScore: Double = 0.0,
        header: String = "Relevant context:"
    ) {
        self.retriever = retriever
        self.topK = topK
        self.minScore = minScore
        self.header = header
    }

    /// Retrieves passages relevant to `text`. Returns an empty array for empty
    /// input or when nothing clears `minScore`.
    public func matches(for text: String) async throws -> [VectorMatch] {
        guard text.isEmpty == false else { return [] }
        return try await retriever.retrieve(
            RetrievalQuery(text: text, topK: topK, minScore: minScore)
        )
    }

    /// Builds grounded `Instructions` by appending passages retrieved for
    /// `topic` to caller-supplied base instructions.
    ///
    /// When nothing is retrieved, the base instructions are returned unchanged.
    public func instructions(
        for topic: String,
        @InstructionsBuilder _ base: () throws -> Instructions
    ) async throws -> Instructions {
        let baseInstructions = try base()
        let matches = try await matches(for: topic)
        let block = RetrievalFormatting.passages(matches, header: header)
        guard block.isEmpty == false else { return baseInstructions }
        return Instructions {
            baseInstructions
            block
        }
    }

    /// Builds a `Prompt` that prepends passages retrieved for `question` to the
    /// question itself.
    ///
    /// When nothing is retrieved, the prompt is just the question.
    public func prompt(augmenting question: String) async throws -> Prompt {
        let matches = try await matches(for: question)
        let block = RetrievalFormatting.passages(matches, header: header)
        guard block.isEmpty == false else { return Prompt(question) }
        return Prompt {
            block
            question
        }
    }
}

extension LanguageModelSession {
    /// Retrieves passages relevant to `question`, prepends them to the prompt,
    /// and responds — a one-call shortcut over ``RetrievalContext/prompt(augmenting:)``.
    ///
    /// - Parameters:
    ///   - question: The user's question.
    ///   - context: The retrieval context used to gather grounding passages.
    ///   - options: Generation options forwarded to the underlying response.
    /// - Returns: The model's response, generated against the grounded prompt.
    nonisolated(nonsending)
    public func respondGrounded(
        to question: String,
        using context: RetrievalContext,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> sending Response<String> {
        let prompt = try await context.prompt(augmenting: question)
        return try await respond(to: prompt, options: options)
    }
}
