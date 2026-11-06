#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import Foundation
import MLX
import MLXLLM
import MLXNN
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

final class GlmCanonicalRequiredToolTests: XCTestCase {
    private static func tool(_ name: String = "file_read", choices: [String] = ["content", "structure"]) -> ToolSpec {
        let mode: [String: any Sendable] = ["type": "string", "enum": choices]
        let path: [String: any Sendable] = ["type": "string"]
        let properties: [String: any Sendable] = ["path": path, "mode": mode]
        let parameters: [String: any Sendable] = ["type": "object", "properties": properties, "required": ["path"]]
        let function: [String: any Sendable] = ["name": name, "parameters": parameters]
        return ["type": "function", "function": function]
    }
    private static func context(_ tools: [ToolSpec], name: String? = nil) throws -> CanonicalRequiredToolContext {
        var additional: [String: any Sendable] = ["tool_choice": "required"]
        if let name { additional["tool_choice_name"] = name }
        return try XCTUnwrap(CanonicalRequiredToolContext(additionalContext: additional, tools: tools))
    }
    func testStableFullSchemaAndSelectorNamespaces() throws {
        let tools = [Self.tool()]
        let required = try Self.context(tools)
        let named = try Self.context(tools, name: "file_read")
        let reordered: ToolSpec = ["function": ["parameters": ["required": ["path"],
            "properties": ["mode": ["enum": ["content", "structure"], "type": "string"] as [String: any Sendable],
                           "path": ["type": "string"] as [String: any Sendable]] as [String: any Sendable],
            "type": "object"] as [String: any Sendable], "name": "file_read"] as [String: any Sendable], "type": "function"]
        XCTAssertEqual(required, try Self.context([reordered]))
        XCTAssertFalse(required.matchesCatalog([Self.tool(choices: ["structure", "content"])]))
        let multi = try Self.context([Self.tool(), Self.tool("file_search")])
        XCTAssertFalse(multi.matchesCatalog([Self.tool("file_search"), Self.tool()]))
        XCTAssertNotEqual(required.requestSalt(ordinarySalt: nil, modelIdentity: "glm-v1"),
                          named.requestSalt(ordinarySalt: nil, modelIdentity: "glm-v1"))
        XCTAssertNotEqual(required.requestSalt(ordinarySalt: nil, modelIdentity: "glm-v1"),
                          required.requestSalt(ordinarySalt: "", modelIdentity: "glm-v1"))
        XCTAssertNotEqual(required.requestSalt(ordinarySalt: "reasoning=on", modelIdentity: "glm-v1"),
                          required.requestSalt(ordinarySalt: "reasoning=off", modelIdentity: "glm-v1"))
        XCTAssertNotEqual(required.requestSalt(ordinarySalt: nil, modelIdentity: "glm-v1"),
                          required.requestSalt(ordinarySalt: nil, modelIdentity: "glm-v2"))
    }
    func testUnknownAndMalformedNamedSelectionsFailClosed() throws {
        let tools = [Self.tool()]
        XCTAssertNil(CanonicalRequiredToolContext(additionalContext: nil, tools: tools))
        XCTAssertNil(CanonicalRequiredToolContext(additionalContext: ["tool_choice": "auto"], tools: tools))
        XCTAssertNil(CanonicalRequiredToolContext(additionalContext: ["tool_choice": "required"], tools: nil))
        XCTAssertNil(CanonicalRequiredToolContext(additionalContext: ["tool_choice": "required"], tools: []))
        XCTAssertNil(CanonicalRequiredToolContext(additionalContext: ["tool_choice": "required"], tools: tools + tools))
        XCTAssertNil(CanonicalRequiredToolContext(additionalContext: ["tool_choice": "required", "tool_choice_name": ""], tools: tools))
        XCTAssertNil(CanonicalRequiredToolContext(additionalContext: ["tool_choice": "required", "tool_choice_name": "other"], tools: tools))
        XCTAssertNil(CanonicalRequiredToolContext(additionalContext: ["tool_choice": "required", "tool_choice_name": "file_read"], tools: tools + [Self.tool("file_search")]))
        XCTAssertNil(CanonicalRequiredToolContext(additionalContext: ["tool_choice": "required", "tool_choice_name": 9], tools: tools))
        XCTAssertEqual(try Self.context(tools, name: "file_read").selectedName, "file_read")
    }
    private static func input(_ scope: CanonicalRequiredToolContext?, tools: [ToolSpec], mask: MLXArray? = nil) -> LMInput {
        LMInput(tokens: MLXArray([Int32(2), 3, 4]), mask: mask, tokenIds: [2, 3, 4],
            cacheScopeSalt: "reasoning=on", cachePrefixTokenCounts: [2],
            cacheRestorePolicy: .freshRequiredToolSelection, toolSchemas: tools,
            canonicalRequiredToolContext: scope)
    }
    func testPreparedCopiesKeepContextWithoutChangingOrdinaryIdentity() throws {
        try MLXMetalTestLock.withLock {
            let tools = [Self.tool()], scope = try Self.context([Self.tool()])
            let prepared = Self.input(scope, tools: tools), unknown = Self.input(nil, tools: tools)
            let parameters = GenerateParameters()
            XCTAssertEqual(computeCacheSalt(for: prepared, parameters: parameters), computeCacheSalt(for: unknown, parameters: parameters))
            XCTAssertEqual(prepared.withToolSchemas(tools).canonicalRequiredToolContext, scope)
            XCTAssertEqual(prepared.withCachePromptIntent(.auxiliary).canonicalRequiredToolContext, scope)
            XCTAssertEqual(prepared.withCacheRestorePolicy(.standard).canonicalRequiredToolContext, scope)
        }
    }
    private final class FixtureBank: Module, WeightedRoutedExpertLayer {
        func callRouted(_ input: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
            XCTFail("Eligibility tests must not execute a model forward")
            return MLXArray.zeros(input.shape, dtype: input.dtype)
        }
    }
    private static func withFixture(_ body: (Glm5Next) throws -> Void) throws {
        let priorAbsorb = Glm5NextIndexerRuntime.absorbMLA, priorGather = Glm5NextIndexerRuntime.gatherSelected
        let priorPool = Glm5NextIndexerRuntime.poolFP32, priorConv = KDAConvRuntime.enabled
        let names = ["MLX_ENABLE_TF32", "VMLX_GLM5_PREFILL_STEP"]
        let priorEnvironment = names.map { name -> String? in getenv(name).map { String(cString: $0) } }
        Glm5NextIndexerRuntime.absorbMLA = true; Glm5NextIndexerRuntime.gatherSelected = true
        Glm5NextIndexerRuntime.poolFP32 = false; KDAConvRuntime.enabled = true
        names.forEach { unsetenv($0) }
        defer {
            Glm5NextIndexerRuntime.absorbMLA = priorAbsorb; Glm5NextIndexerRuntime.gatherSelected = priorGather
            Glm5NextIndexerRuntime.poolFP32 = priorPool; KDAConvRuntime.enabled = priorConv
            for (name, prior) in zip(names, priorEnvironment) {
                if let prior { setenv(name, prior, 1) } else { unsetenv(name) }
            }
        }
        let config = try JSONDecoder().decode(Glm5NextConfiguration.self, from: Data(tinyJSON.utf8))
        var banks: [Int: any WeightedRoutedExpertLayer] = [:]
        for layer in 0..<config.textConfig.numHiddenLayers where config.textConfig.mlpLayerTypes[layer] == .sparse {
            banks[layer] = FixtureBank()
        }
        let exclusions = Set(banks.keys.flatMap { layer in
            ["gate_proj", "up_proj", "down_proj"].flatMap { role in
                ["tq2_packed", "tq2_scales"].map { "model.layers.\(layer).mlp.switch_mlp.\(role).\($0)" }
            }
        })
        let model = try Glm5Next(config, requesting: [.text], routedExperts: banks, customRoutedTensorNames: exclusions)
        let embedding = QuantizedEmbedding(
            weight: MLXArray.zeros([128, 16], dtype: .uint32),
            scales: MLXArray.ones([128, 1], dtype: .bfloat16),
            biases: MLXArray.zeros([128, 1], dtype: .bfloat16), groupSize: 64, bits: 8)
        model.languageModel.update(modules: ModuleChildren.unflattened([("embed_tokens", embedding)]))
        try body(model)
    }
    func testPackedEmbeddingEligibilityAndRequestedCompileParity() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFixture { model in
                var parameters = GenerateParameters(prefillStepSize: 512)
                let embedding = try XCTUnwrap(model.languageModel.embedTokens as? QuantizedEmbedding)
                XCTAssertEqual(embedding.weight.dtype, .uint32)
                XCTAssertEqual(model.canonicalRequiredToolChunkSize(parameters: parameters), 512)
                parameters.enableCompiledDecode = true
                XCTAssertEqual(model.canonicalRequiredToolChunkSize(parameters: parameters), 512)
                embedding.outputDType = .float16
                XCTAssertNil(model.canonicalRequiredToolChunkSize(parameters: parameters))
                embedding.outputDType = .bfloat16
                XCTAssertEqual(model.canonicalRequiredToolChunkSize(parameters: parameters), 512)
                var changed = parameters; changed.prefillStepSize = 256
                XCTAssertNil(model.canonicalRequiredToolChunkSize(parameters: changed))
                changed = parameters; changed.kvBits = 4
                XCTAssertNil(model.canonicalRequiredToolChunkSize(parameters: changed))
                changed = parameters; changed.maxKVSize = 16
                XCTAssertNil(model.canonicalRequiredToolChunkSize(parameters: changed))
                setenv("MLX_ENABLE_TF32", "0", 1)
                XCTAssertNil(model.canonicalRequiredToolChunkSize(parameters: parameters))
            }
        }
    }
    func testMasksAndUnqualifiedPreparedInputsFailClosed() throws {
        try MLXMetalTestLock.withLock {
            try Self.withFixture { model in
                let tools = [Self.tool()], scope = try Self.context([Self.tool()])
                let parameters = GenerateParameters(prefillStepSize: 512), cache = model.newCache(parameters: parameters)
                let prepared = Self.input(scope, tools: tools)
                func salt(_ input: LMInput) -> String? {
                    canonicalRequiredToolSalt(input: input, model: model, parameters: parameters, cache: cache, ordinarySalt: "reasoning=on")
                }
                XCTAssertNotNil(salt(prepared))
                // Even an all-ones caller mask is outside the qualified processor contract.
                for mask in [MLXArray([Int32(1), 1, 1]), MLXArray([Int32(1), 0, 1])] {
                    XCTAssertNil(salt(Self.input(scope, tools: tools, mask: mask)))
                }
                XCTAssertNil(salt(Self.input(nil, tools: tools)))
                XCTAssertNil(salt(prepared.withCacheRestorePolicy(.standard)))
                XCTAssertNil(salt(prepared.withCachePromptIntent(.auxiliary)))
                XCTAssertNil(salt(prepared.withToolSchemas([Self.tool(choices: ["structure", "content"])])))
            }
        }
    }
    // Same bounded construction geometry used by existing Glm5NextConstructionTests;
    // self-contained so the focused target does not need the whole large test file.
    private static let tinyJSON = #"""
        {"model_type":"glm5_next","image_token_id":9,"video_token_id":10,
         "text_config":{"model_type":"glm5_next_text","hidden_size":64,
           "num_hidden_layers":4,"intermediate_size":128,"num_attention_heads":4,
           "num_key_value_heads":4,"vocab_size":128,"rms_norm_eps":1e-05,
           "max_position_embeddings":4096,"kv_lora_rank":16,"q_lora_rank":32,
           "qk_nope_head_dim":16,"qk_rope_head_dim":0,"v_head_dim":16,"mla_use_nope":true,
           "n_routed_experts":8,"n_shared_experts":1,"num_experts_per_tok":2,
           "moe_intermediate_size":32,"first_k_dense_replace":2,"scoring_func":"sigmoid",
           "topk_method":"noaux_tc","routed_scaling_factor":2.5,"norm_topk_prob":true,
           "n_group":1,"topk_group":1,"mhc":true,"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,
           "index_head_dim":16,"index_n_heads":2,"index_topk":2048,"index_kpool":4,
           "index_kpool_compress":true,"index_kpool_always_select_tail":true,
           "num_nextn_predict_layers":0,"swiglu_limit":10.0,"tie_word_embeddings":false,
           "linear_attn_config":{"num_heads":4,"gate_lower_bound":-5.0,"head_dim":16,
             "short_conv_kernel_size":4,"kda_layers":[0,2,3],"full_attn_layers":[1]},
           "layer_types":["linear_attention","deepseek_sparse_attention","linear_attention","linear_attention"],
           "mlp_layer_types":["dense","dense","sparse","sparse"]}}
        """#
}
