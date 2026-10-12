// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
@_spi(GemmaEncoder) import MLXVLM
import Testing

/// Verifies that the `@_spi(GemmaEncoder)` `textHiddenStates(_:)` on the Gemma 4 VLMs returns
/// exactly the hidden states the LM head consumes, so a client head can read them without a
/// second, text-only copy of the model.
///
/// Mirrors `Gemma4EncoderAccessTests` (#530): imports `MLXVLM` without `@testable`, opting in
/// via `@_spi(GemmaEncoder)` as a client module would, so it fails to compile if the method is
/// not reachable from outside the package. The tied LM head and the logit softcap are applied
/// below from public API only (`parameters()` and the configuration). Tiny randomly-initialized
/// models; no weights are downloaded.
struct Gemma4VLMHiddenStatesAccessTests {

    enum Variant: String, CaseIterable, CustomTestStringConvertible {
        case gemma4, gemma4Unified
        var testDescription: String { rawValue }
    }

    /// The SPI output next to everything a client needs to rebuild the logits from it.
    private struct Run {
        let hidden: MLXArray
        let logits: MLXArray
        let parameters: [String: MLXArray]
        let hiddenSize: Int
        let softcap: Float?
    }

    /// E2B-shaped text config: per-layer inputs and KV-shared tail layers are on, as in
    /// `gemma-4-e2b-it`.
    private static let gemma4JSON = """
        {
          "text_config": {
            "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "global_head_dim": 32,
            "vocab_size": 64, "vocab_size_per_layer_input": 64,
            "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 8,
            "sliding_window": 8, "sliding_window_pattern": 2,
            "max_position_embeddings": 512
          },
          "vision_config": {
            "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
            "num_attention_heads": 2, "head_dim": 8, "position_embedding_size": 64
          }
        }
        """

    private static let gemma4UnifiedJSON = """
        {
          "model_type": "gemma4_unified", "vocab_size": 64,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
            "vocab_size": 64, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "sliding_window": 8, "sliding_window_pattern": 2,
            "max_position_embeddings": 512
          },
          "vision_config": null,
          "audio_config": null
        }
        """

    private static func run(_ variant: Variant, tokens: MLXArray) throws -> Run {
        switch variant {
        case .gemma4:
            let config = try JSONDecoder().decode(
                Gemma4Configuration.self, from: Data(gemma4JSON.utf8))
            let model = Gemma4(config)
            return Run(
                hidden: model.textHiddenStates(tokens), logits: model(tokens, cache: nil),
                parameters: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                hiddenSize: config.textConfiguration.hiddenSize,
                softcap: config.textConfiguration.finalLogitSoftcapping)
        case .gemma4Unified:
            let config = try JSONDecoder().decode(
                Gemma4UnifiedConfiguration.self, from: Data(gemma4UnifiedJSON.utf8))
            let model = Gemma4Unified(config)
            return Run(
                hidden: model.textHiddenStates(tokens), logits: model(tokens, cache: nil),
                parameters: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                hiddenSize: config.textConfiguration.hiddenSize,
                softcap: config.textConfiguration.finalLogitSoftcapping)
        }
    }

    /// The tied LM head plus the logit softcap, applied to `textHiddenStates`, must give the
    /// model's own logits. A pre-norm state, the input embeddings, or a state from another
    /// path would not.
    @Test(arguments: Variant.allCases)
    func textHiddenStatesAreTheLMHeadInput(_ variant: Variant) throws {
        let tokens = MLXArray((0 ..< 8).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, 8)
        let run = try Self.run(variant, tokens: tokens)

        // Gemma 4 ties the LM head to the input embedding.
        #expect(run.parameters["language_model.lm_head.weight"] == nil)
        let embedding = try #require(run.parameters["language_model.model.embed_tokens.weight"])

        var expected = matmul(run.hidden, embedding.T)
        if let cap = run.softcap, cap > 0 {
            expected = tanh(expected / cap) * cap
        }
        eval(expected, run.logits)

        #expect(run.hidden.shape == [1, 8, run.hiddenSize])
        #expect(expected.shape == run.logits.shape)
        #expect(allClose(expected, run.logits, atol: 1e-5).item(Bool.self))
    }
}
