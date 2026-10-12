// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXScriptedLM
import MLXVLM
import Testing

/// `loadModelContainer(from:using:)` tries `VLMModelFactory` before `LLMModelFactory`.
/// The fixture is a `gemma3` checkpoint that both factories register, with text weights
/// only, so the VLM load fails and the LLM load succeeds.
struct ModelFactoryFallbackTests {

    @Test("A multimodal checkpoint the VLM factory fails to load does not fall back to text-only")
    func visionCheckpointSurfacesVLMError() async throws {
        let directory = try makeGemma3Checkpoint(visionConfiguration: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        await #expect(throws: UpdateError.self) {
            _ = try await loadModelContainer(from: directory, using: TokenizerLoaderStub())
        }
    }

    @Test("A text-only export of a multimodal model still loads through the LLM factory")
    func textOnlyCheckpointFallsBackToLLM() async throws {
        let directory = try makeGemma3Checkpoint(visionConfiguration: false)
        defer { try? FileManager.default.removeItem(at: directory) }

        let container = try await loadModelContainer(from: directory, using: TokenizerLoaderStub())
        let isTextModel = await container.perform { $0.model is Gemma3TextModel }
        #expect(isTextModel)
    }

    private func makeGemma3Checkpoint(visionConfiguration: Bool) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let textConfig = """
            {
              "model_type": "gemma3_text", "hidden_size": 8, "num_hidden_layers": 1,
              "intermediate_size": 16, "num_attention_heads": 1, "num_key_value_heads": 1,
              "head_dim": 8, "vocab_size": 16, "sliding_window": 8
            }
            """
        let visionConfig = """
            {
              "model_type": "siglip_vision_model", "num_hidden_layers": 1, "hidden_size": 8,
              "intermediate_size": 16, "num_attention_heads": 1, "patch_size": 4,
              "image_size": 8
            }
            """
        let config = """
            {
              "model_type": "gemma3", "mm_tokens_per_image": 4,
              "text_config": \(textConfig)\(visionConfiguration ? ", \"vision_config\": \(visionConfig)" : "")
            }
            """
        try Data(config.utf8).write(to: directory.appendingPathComponent("config.json"))

        let model = Gemma3TextModel(
            try JSONDecoder().decode(Gemma3TextConfiguration.self, from: Data(textConfig.utf8)))
        eval(model)
        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map {
                ("language_model.\($0.0)", $0.1)
            })
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    private struct TokenizerLoaderStub: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer {
            PseudoWordTokenizer()
        }
    }
}
