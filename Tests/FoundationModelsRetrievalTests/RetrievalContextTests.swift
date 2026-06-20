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

import Testing
import FoundationModels
@testable import FoundationModelsRetrieval

@Suite struct RetrievalContextTests {
    private let sample: [VectorMatch] = [
        .make(id: "1", text: "Full-time employees accrue 20 days.", score: 0.9),
    ]

    @Test func retrievesMatchesForText() async throws {
        let context = RetrievalContext(retriever: StubRetriever(matches: sample))
        let matches = try await context.matches(for: "vacation")
        #expect(matches.map(\.record.id) == ["1"])
    }

    @Test func emptyTextRetrievesNothing() async throws {
        let context = RetrievalContext(retriever: StubRetriever(matches: sample))
        #expect(try await context.matches(for: "").isEmpty)
    }

    @Test func buildsPromptAugmentingQuestion() async throws {
        // `Prompt`/`Instructions` expose no public content accessor, so these
        // are smoke tests: building a grounded prompt must not throw. Passage
        // rendering itself is covered by `RetrievalFormattingTests`.
        let context = RetrievalContext(retriever: StubRetriever(matches: sample))
        _ = try await context.prompt(augmenting: "How many vacation days?")
    }

    @Test func buildsPromptWithoutMatches() async throws {
        let context = RetrievalContext(retriever: StubRetriever(matches: []))
        _ = try await context.prompt(augmenting: "Unrelated question")
    }

    @Test func buildsGroundedInstructions() async throws {
        let context = RetrievalContext(retriever: StubRetriever(matches: sample))
        _ = try await context.instructions(for: "vacation policy") {
            Instructions("You are an HR assistant.")
        }
    }
}
