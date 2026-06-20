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
import Synchronization
@testable import FoundationModelsRetrieval

@Suite struct RetrievalToolTests {
    private let sample: [VectorMatch] = [
        .make(id: "1", text: "Refunds are issued within 30 days.", score: 0.9),
        .make(id: "2", text: "Shipping is free over $50.", score: 0.7),
    ]

    @Test func rendersNumberedPassagesWithHeader() async throws {
        let tool = RetrievalTool(retriever: StubRetriever(matches: sample), header: "Relevant passages:")
        let output = try await tool.call(arguments: RetrievalArguments(query: "refunds"))
        #expect(output == """
        Relevant passages:
        [1] Refunds are issued within 30 days.
        [2] Shipping is free over $50.
        """)
    }

    @Test func exposesRawMatchesAfterCall() async throws {
        let tool = RetrievalTool(retriever: StubRetriever(matches: sample))
        #expect(tool.lastMatches.isEmpty)
        _ = try await tool.call(arguments: RetrievalArguments(query: "refunds"))
        #expect(tool.lastMatches.map(\.record.id) == ["1", "2"])
    }

    @Test func invokesOnResultsCallback() async throws {
        let captured = Mutex<[VectorMatch]>([])
        let tool = RetrievalTool(
            retriever: StubRetriever(matches: sample),
            onResults: { matches in captured.withLock { $0 = matches } }
        )
        _ = try await tool.call(arguments: RetrievalArguments(query: "refunds"))
        #expect(captured.withLock { $0 }.count == 2)
    }

    @Test func honorsTopKOverride() async throws {
        let tool = RetrievalTool(retriever: StubRetriever(matches: sample), defaultTopK: 5)
        _ = try await tool.call(arguments: RetrievalArguments(query: "refunds", topK: 1))
        #expect(tool.lastMatches.count == 1)
    }

    @Test func returnsMessageWhenNothingFound() async throws {
        let tool = RetrievalTool(retriever: StubRetriever(matches: []))
        let output = try await tool.call(arguments: RetrievalArguments(query: "nothing"))
        #expect(output == "No relevant passages were found.")
    }
}

@Suite struct RetrievalFormattingTests {
    @Test func emptyMatchesRenderEmptyString() {
        #expect(RetrievalFormatting.passages([], header: "Header").isEmpty)
    }

    @Test func omitsHeaderWhenEmpty() {
        let block = RetrievalFormatting.passages([.make(id: "1", text: "x", score: 1)], header: "")
        #expect(block == "[1] x")
    }
}
