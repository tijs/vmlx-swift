import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import XCTest
@testable import MLXLLM

final class Inkling2LoopTests: XCTestCase {
    func configurationData() throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "model_type":"inkling2_loop", "text_config":[
                "model_type":"inkling2_loop_text", "hidden_size":8, "num_hidden_layers":4,
                "vocab_size":32,"unpadded_vocab_size":31,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
                "swa_num_attention_heads":2,"swa_num_key_value_heads":1,"swa_head_dim":4,
                "d_rel":2,"rel_extent":4,"sliding_window_size":3,"sconv_kernel_size":2,
                "intermediate_size":8,"n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":0,
                "n_layers_per_attn_res_block":1,"local_layer_ids":[0,2],"rms_norm_eps":1e-6,
                "route_scale":2.0,"qk_l2_clip_threshold":4.0,"rel_l2_clip_threshold":2.0,
                "mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"final_logit_softcapping":30.0,
                "log_scaling_alpha":0.1,"log_scaling_n_floor":8,
                "attn_res_normalize_values":true,"use_value_norm_no_gamma":true,
                "loop_config":["core_start":1,"core_end":3,"num_passes":3,"gate_rank":2,"is_varloop":false]
            ]
        ])
    }
    func config() throws -> Inkling2LoopConfiguration { try JSONDecoder().decode(Inkling2LoopConfiguration.self,from:configurationData()) }
    func difference(_ a:MLXArray,_ b:MLXArray)->Float { max(abs(a.asType(.float32)-b.asType(.float32))).item(Float.self) }
    // Generated from the pinned authored Python primitives and separately
    // cross-checked against NumPy; fixture is not a Swift self-oracle.
    func testAuthoredMessageGateAndAttentionResidualGolden() throws {
        try Device.withDefaultDevice(.cpu) {
            let url=try XCTUnwrap(Bundle.module.url(forResource:"Inkling2Primitives",withExtension:"json"))
            let fixture=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
            func array(_ value:Any)->MLXArray {
                func dimensions(_ value:Any)->[Int] {
                    guard let a=value as? [Any] else { return [] }
                    return [a.count]+(a.first.map(dimensions) ?? [])
                }
                func values(_ value:Any)->[Float] {
                    if let a=value as? [Any] { return a.flatMap(values) }
                    return [(value as! NSNumber).floatValue]
                }
                return MLXArray(values(value),dimensions(value))
            }
            var root=try JSONSerialization.jsonObject(with:configurationData()) as! [String:Any]
            var text=root["text_config"] as! [String:Any];text["hidden_size"]=4;root["text_config"]=text
            let cfg=try JSONDecoder().decode(Inkling2LoopConfiguration.self,from:JSONSerialization.data(withJSONObject:root))
            let gate=InklingMessageGate(cfg.text)
            try gate.update(parameters:ModuleParameters.unflattened([
                "down_proj.weight":array(fixture["down"]!),
                "forget_gate_proj.weight":array(fixture["forget"]!),
                "message_gate_proj.weight":array(fixture["gain"]!)
            ]),verify:.all)
            let got=gate(array(fixture["hidden"]!),array(fixture["message"]!))
            XCTAssertLessThan(difference(got,array(fixture["gate_expected"]!)),2e-6)
            let expected=fixture["residual_expected"] as! [String:Any]
            for normalized in [false,true] {
                let got=inklingAttentionResidual(array(fixture["states"]!),query:array(fixture["query"]!),normalizeValues:normalized)
                XCTAssertLessThan(difference(got,array(expected[String(normalized)]!)),2e-6)
            }
        }
    }
    func testScheduleAndRejectMalformedLoop() throws {
        let c=try config()
        XCTAssertEqual(c.text.schedule,[0,1,2,1,2,1,2,3])
        var root=try JSONSerialization.jsonObject(with:configurationData()) as! [String:Any]
        var text=root["text_config"] as! [String:Any]
        let loop=text["loop_config"] as! [String:Any]
        for (key,value) in [("num_passes",2 as Any),("core_start",-1 as Any),("is_varloop",true as Any),("gate_rank",0 as Any)] {
            var mutated=loop;mutated[key]=value;text["loop_config"]=mutated;root["text_config"]=text
            XCTAssertThrowsError(try JSONDecoder().decode(Inkling2LoopConfiguration.self,from:JSONSerialization.data(withJSONObject:root)))
        }
    }
    func testCacheCopiesAllCompanionStatesAndCannotTrim() throws {
        Device.withDefaultDevice(.cpu) {
            let c=Inkling2VisitCache()
            _=c.update(keys:.ones([1,1,2,4]),values:.ones([1,1,2,4]))
            for i in 0..<4 { c.convolutions[i] = .ones([1,1,4]) * Float(i+1) }
            let copy=c.copy() as! Inkling2VisitCache
            c.convolutions[0] = .zeros([1,1,4])
            XCTAssertEqual(copy.offset,2);XCTAssertEqual(copy.state.count,6)
            XCTAssertEqual(copy.convolutions[0]!.sum().item(Float.self),4)
            XCTAssertFalse(copy.isTrimmable);XCTAssertEqual(copy.trim(1),0)
        }
    }
    func testShortConvolutionCachedChunksMatchFull() throws {
        try Device.withDefaultDevice(.cpu) {
            let t=try config().text
            let conv=InklingShortConvolution(channels:8,kernel:2,index:0)
            try conv.update(parameters:ModuleParameters.unflattened(["conv.weight":MLXArray(Array(repeating:Float(0.25),count:16),[8,2,1])]),verify:.all)
            let x=MLXArray((0..<40).map { Float($0)/40 },[1,5,8])
            let full=conv(x,nil)
            let cache=Inkling2VisitCache()
            let parts=[conv(x[0...,0..<2,0...],cache),conv(x[0...,2..<5,0...],cache)]
            XCTAssertLessThan(difference(full,concatenated(parts,axis:1)),1e-6)
            let dynamic=InklingDynamicConvolution(t,channels:4,heads:1,index:1)
            let weights=Dictionary(uniqueKeysWithValues:dynamic.parameters().flattened().map { ($0.0,MLXArray.ones($0.1.shape)*0.125) })
            try dynamic.update(parameters:ModuleParameters.unflattened(weights),verify:.all)
            let v=x[0...,0...,0..<4]
            let expected=dynamic(v,input:x,cache:nil)
            let c=Inkling2VisitCache()
            let a=dynamic(v[0...,0..<2,0...],input:x[0...,0..<2,0...],cache:c)
            let b=dynamic(v[0...,2..<5,0...],input:x[0...,2..<5,0...],cache:c)
            XCTAssertLessThan(difference(expected,concatenated([a,b],axis:1)),1e-6)
        }
    }
    func testStrictTensorInventoryAndTinyCachedModel() throws {
        try Device.withDefaultDevice(.cpu) {
            let m=Inkling2LoopModel(try config())
            let p=Dictionary(uniqueKeysWithValues:m.parameters().flattened())
            var weights:[String:MLXArray]=[:]
            for (key,a) in p {
                let values=(0..<a.size).map { Float(($0 % 11)-5)*0.003 }
                weights[key] = key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") ? .ones(a.shape):MLXArray(values,a.shape)
            }
            let strict=try m.sanitizeWeights(weights,metadata:[:])
            XCTAssertEqual(strict.count,weights.count)
            var missing=weights;missing["language_model.model.embed_tokens.weight"]=nil
            XCTAssertThrowsError(try m.sanitizeWeights(missing,metadata:[:]))
            var extra=weights;extra["unknown.weight"] = .zeros([1])
            XCTAssertThrowsError(try m.sanitizeWeights(extra,metadata:[:]))
            try m.update(parameters:ModuleParameters.unflattened(strict),verify:.all)
            let tokens=MLXArray([Int32(1),2,3,4,5],[1,5])
            let full=m(tokens,cache:nil)
            struct Reference: Decodable { let expected_logits: [[[Float]]] }
            let referenceURL = try XCTUnwrap(Bundle.module.url(forResource: "InklingTinyModel", withExtension: "json"))
            let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: referenceURL))
            let referenceLogits = MLXArray(reference.expected_logits.flatMap { $0.flatMap { $0 } }, [1,5,31])
            XCTAssertLessThan(difference(full, referenceLogits), 1e-6, "full model must match authored Python, not only its own cache path")
            let cache=m.newCache(parameters:nil)
            XCTAssertEqual(cache.count,8);XCTAssertEqual(Set(cache.map { ObjectIdentifier($0 as AnyObject) }).count,8)
            let a=m(tokens[0...,0..<2],cache:cache)
            let b=m(tokens[0...,2..<5],cache:cache)
            XCTAssertLessThan(difference(full,concatenated([a,b],axis:1)),2e-5)
            XCTAssertTrue(cache.allSatisfy { $0.offset==5 && $0.state.count==6 })
        }
    }
    func testPublishedCheckpointMetadataMatchesEveryNativeParameter() throws {
        try Device.withDefaultDevice(.cpu) {
            struct Tensor: Decodable { let shape: [Int]; let dtype: String }
            struct Inventory: Decodable {
                let config: Inkling2LoopConfiguration
                let tensors: [String: Tensor]
            }
            let url = try XCTUnwrap(Bundle.module.url(forResource: "InklingPublishedInventory", withExtension: "json"))
            let fixture = try JSONDecoder().decode(Inventory.self, from: Data(contentsOf: url))
            let model = Inkling2LoopModel(fixture.config)
            let types: [String: DType] = ["BF16": .bfloat16, "F32": .float32, "U8": .uint8, "U32": .uint32]
            // Shape/dtype-only lazy placeholders. Do not evaluate these arrays:
            // this test checks the complete checkpoint contract, not weights.
            var weights: [String: MLXArray] = [:]
            for (key, tensor) in fixture.tensors {
                weights[key] = .zeros(tensor.shape, dtype: try XCTUnwrap(types[tensor.dtype]))
            }
            XCTAssertEqual(weights.count, 1271)
            let sanitized = try model.sanitizeWeights(weights, metadata: [:])
            XCTAssertEqual(sanitized.count, 849)
            XCTAssertEqual(weights.keys.filter { $0.hasPrefix("vision_tower.") }.count, 422)
            XCTAssertEqual(model.newCache(parameters: nil).count, 48)
            for (key, tensor) in sanitized {
                XCTAssertEqual(tensor.dtype, weights[key]!.dtype, key)
                XCTAssertEqual(tensor.shape, weights[key]!.shape, key)
            }
            try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
            let router = "language_model.model.layers.0.mlp.gate_weight"
            XCTAssertEqual(Dictionary(uniqueKeysWithValues: model.parameters().flattened())[router]!.dtype, .bfloat16)
            var missing = weights
            missing["language_model.model.layers.23.mlp.switch_mlp.down_proj.weight"] = nil
            XCTAssertThrowsError(try model.sanitizeWeights(missing, metadata: [:]))
            weights["mtp.draft.weight"] = .zeros([1])
            XCTAssertThrowsError(try model.sanitizeWeights(weights, metadata: [:]))
        }
    }

    func testAuthoredMixedBF16MetalModelAndChunkedCache() throws {
        guard ProcessInfo.processInfo.environment["VMLX_INKLING_METAL_FIXTURE"] == "1" else {
            throw XCTSkip("Opt in only in an exclusive safe Metal window")
        }
        try Device.withDefaultDevice(.gpu) {
            let model = Inkling2LoopModel(try config())
            var weights: [String: MLXArray] = [:]
            for (key, array) in model.parameters().flattened() {
                let values = (0..<array.size).map { Float(($0 % 11)-5)*0.003 }
                let fp32 = key.hasSuffix(".query") || key.hasSuffix(".e_score_correction_bias") || key.hasSuffix(".weight_scale_2")
                let tensor: MLXArray = key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") ? .ones(array.shape) : MLXArray(values, array.shape)
                weights[key] = tensor.asType(fp32 ? .float32 : .bfloat16)
            }
            try model.update(parameters: ModuleParameters.unflattened(model.sanitizeWeights(weights, metadata: [:])), verify: .all)
            struct Reference: Decodable { let expected_logits: [[[Float]]] }
            let url = try XCTUnwrap(Bundle.module.url(forResource: "InklingTinyBF16Model", withExtension: "json"))
            let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
            let expected = MLXArray(reference.expected_logits.flatMap { $0.flatMap { $0 } }, [1,5,31])
            let tokens = MLXArray([Int32(1),2,3,4,5], [1,5])
            let full = model(tokens, cache: nil)
            XCTAssertLessThan(difference(full, expected), 2e-5)
            XCTAssertEqual(argMax(full, axis: -1).asArray(Int32.self), argMax(expected, axis: -1).asArray(Int32.self))
            let cache = model.newCache(parameters: nil)
            let a = model(tokens[0...,0..<2], cache: cache)
            let b = model(tokens[0...,2..<5], cache: cache)
            XCTAssertLessThan(difference(full, concatenated([a,b], axis: 1)), 2e-5)
            XCTAssertTrue(cache.allSatisfy { $0.offset == 5 && $0.state.count == 6 })
        }
    }

    func testAuthoredBF16RouterStorageIsPreserved() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Inkling2LoopModel(try config())
            var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            let routers = weights.keys.filter { $0.hasSuffix(".gate_weight") }
            XCTAssertEqual(routers.count, 4)
            for key in routers { weights[key] = weights[key]!.asType(.bfloat16) }
            let sanitized = try model.sanitizeWeights(weights, metadata: [:])
            for key in routers { XCTAssertEqual(sanitized[key]!.dtype, .bfloat16) }
            try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
            let loaded = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            for key in routers { XCTAssertEqual(loaded[key]!.dtype, .bfloat16) }
            weights[routers[0]] = weights[routers[0]]!.asType(.float16)
            XCTAssertThrowsError(try model.sanitizeWeights(weights, metadata: [:]))
        }
    }

    func testNativeNVFP4OuterScaleMatchesAuthoredPackedReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let url = try XCTUnwrap(Bundle.module.url(forResource: "InklingNVFP4", withExtension: "json"))
            let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
            func shape(_ x: Any) -> [Int] {
                guard let a = x as? [Any] else { return [] }
                return [a.count] + (a.first.map(shape) ?? [])
            }
            func numbers(_ x: Any) -> [NSNumber] {
                if let a = x as? [Any] { return a.flatMap(numbers) }
                return [x as! NSNumber]
            }
            func floats(_ key: String) -> MLXArray {
                MLXArray(numbers(fixture[key]!).map { $0.floatValue }, shape(fixture[key]!))
            }
            let projection = InklingExpertProjection(input: 32, output: 16, experts: 3,
                quant: .init(groupSize: 16, bits: 4, mode: "nvfp4"))
            let weights = MLXArray(numbers(fixture["weight"]!).map { $0.uint32Value }, shape(fixture["weight"]!))
            let scales = MLXArray(numbers(fixture["scales"]!).map { $0.uint8Value }, shape(fixture["scales"]!))
            try projection.update(parameters: ModuleParameters.unflattened([
                "weight": weights, "scales": scales, "weight_scale_2": floats("outer")]), verify: .all)
            let indices = MLXArray(numbers(fixture["indices"]!).map { $0.uint32Value }, shape(fixture["indices"]!))
            let output = projection(floats("input").asType(.bfloat16), indices)
            XCTAssertEqual(output.dtype, .bfloat16)
            XCTAssertLessThan(difference(output, floats("expected")), 1e-5)
        }
    }

    func testAuthoredResidualReciprocalPreservesBF16RoutingInput() throws {
        guard ProcessInfo.processInfo.environment["VMLX_INKLING_METAL_FIXTURE"] == "1" else {
            throw XCTSkip("Opt in only in an exclusive safe Metal window")
        }
        try Device.withDefaultDevice(.gpu) {
            // Independent authored Python layer2 golden. With 2304 channels,
            // division by 48 changes one BF16 value relative to multiplying
            // the correctly rounded reciprocal; the routed layer amplifies it.
            let url = try XCTUnwrap(Bundle.module.url(forResource: "InklingResidualRounding", withExtension: "safetensors"))
            let f = try loadArrays(url: url)
            let states = try XCTUnwrap(f["states"]), query = try XCTUnwrap(f["query"])
            let expected = try XCTUnwrap(f["expected_normalized"])
            let weight = try XCTUnwrap(f["norm_weight"])
            let mixed = inklingAttentionResidual(states, query: query, normalizeValues: true)
            let normalized = MLXFast.rmsNorm(mixed.squeezed(axis: -2), weight: weight, eps: 1e-6).asType(.bfloat16)
            XCTAssertEqual(difference(normalized, expected), 0, "The actual BF16 input to routing must match the independent reference")
            let keys = MLXFast.rmsNorm(states, weight: .ones([states.dim(-1)]), eps: 1e-6)
            let oldScores = matmul(keys, query.reshaped([-1, states.dim(-1)]).T) / sqrt(Float(states.dim(-1)))
            let oldMixed = matmul(softmax(oldScores, axis: -2).swappedAxes(-1, -2), keys)
            let oldNormalized = MLXFast.rmsNorm(oldMixed.squeezed(axis: -2), weight: weight, eps: 1e-6).asType(.bfloat16)
            XCTAssertGreaterThan(difference(oldNormalized, expected), 0, "Golden must expose the original rounding defect")
            if let path = ProcessInfo.processInfo.environment["VMLX_INKLING_RESIDUAL_DIAGNOSTICS"] {
                let scores = matmul(keys, query.reshaped([-1, states.dim(-1)]).T) * Float(pow(Double(states.dim(-1)), -0.5))
                let weights = softmax(scores, axis: -2).swappedAxes(-1, -2)
                try save(arrays: ["ones-keys": keys, "ones-scores": scores,
                    "ones-weights": weights, "ones-result": mixed], url: URL(fileURLWithPath: path))
            }

        }
    }

    func testVirtualShortConvolutionMatchesIndependentFP32HistoryOracle() throws {
        guard ProcessInfo.processInfo.environment["VMLX_INKLING_METAL_FIXTURE"] == "1" else {
            throw XCTSkip("Opt in only in an exclusive safe Metal window")
        }
        struct Case: Decodable {
            let length: Int
            let input, history, expectedOutput, expectedHistory: [Float]
        }
        struct Fixture: Decodable {
            let channels, width: Int
            let weights: [Float]
            let cases: [Case]
        }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "InklingShortConvolution", withExtension: "json"))
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        try Device.withDefaultDevice(.gpu) {
            let conv = InklingShortConvolution(channels: f.channels, kernel: f.width, index: 0)
            let weight = MLXArray(f.weights, [f.channels,f.width,1]).asType(.bfloat16)
            try conv.update(parameters: ModuleParameters.unflattened(["conv.weight": weight]), verify: .all)
            for item in f.cases {
                let input = MLXArray(item.input, [1,item.length,f.channels]).asType(.bfloat16)
                let history = MLXArray(item.history, [1,f.width-1,f.channels])
                let expected = MLXArray(item.expectedOutput, input.shape)
                let fused = Inkling2InferenceKernels.virtualShortConvolution(
                    state: history, weight: weight, input: input, width: f.width)
                XCTAssertEqual(difference(fused, expected), 0, "length=\(item.length)")
                let stridedInput = contiguous(input.swappedAxes(1,2)).swappedAxes(1,2)
                let stridedHistory = contiguous(history.swappedAxes(1,2)).swappedAxes(1,2)
                let stridedWeight = contiguous(weight.swappedAxes(0,1)).swappedAxes(0,1)
                let strided = Inkling2InferenceKernels.virtualShortConvolution(
                    state: stridedHistory, weight: stridedWeight, input: stridedInput, width: f.width)
                XCTAssertEqual(difference(strided, expected), 0, "strided length=\(item.length)")

                let cache = Inkling2VisitCache()
                cache.convolutions[0] = history
                let result = conv(input, cache)
                XCTAssertEqual(difference(result, expected), 0, "integration length=\(item.length)")
                XCTAssertEqual(difference(cache.convolutions[0]!, MLXArray(item.expectedHistory, history.shape)), 0)
                XCTAssertEqual(cache.convolutions[0]!.dtype, .float32)
            }
        }
    }

    func testSortedExpertPrefillRestoresIndependentRouteOrder() throws {
        guard ProcessInfo.processInfo.environment["VMLX_INKLING_METAL_FIXTURE"] == "1" else {
            throw XCTSkip("Opt in only in an exclusive safe Metal window")
        }
        try Device.withDefaultDevice(.gpu) {
            let c = try config(), t = c.text, tokens = 40
            let mlp = InklingSwitchMLP(c, layer: 0)
            let firstFeature = MLXArray((0..<(t.experts*t.intermediate*t.hidden)).map {
                Float($0 % t.hidden == 0 ? 1 : 0)
            }, [t.experts,t.intermediate,t.hidden])
            let down = MLXArray((0..<(t.experts*t.hidden*t.intermediate)).map {
                Float($0 / (t.hidden*t.intermediate) + 1) / Float(t.intermediate)
            }, [t.experts,t.hidden,t.intermediate])
            try mlp.update(parameters: ModuleParameters.unflattened([
                "gate_proj.weight":firstFeature, "up_proj.weight":firstFeature, "down_proj.weight":down
            ]), verify: .all)
            let values = (0..<(tokens*t.hidden)).map { Float($0/t.hidden-20)/32 + Float($0%t.hidden)/128 }
            let selected = (0..<(tokens*t.topK)).map { UInt32((3*($0/t.topK)+2*($0%t.topK))%t.experts) }
            let input = MLXArray(values, [1,tokens,t.hidden])
            let indices = MLXArray(selected, [1,tokens,t.topK])
            var expected: [Float] = []
            for token in 0..<tokens {
                // Each gate/up row selects x[0]. Every down row averages the
                // identical activations then scales by the selected expert+1.
                // Calculate the independent scalar formula in Float64.
                let x = Double(values[token*t.hidden])
                let activation = tanh(x/Double(t.gateCap))*Double(t.gateCap)
                    / (1+exp(-x)) * tanh(x/Double(t.upCap))*Double(t.upCap)
                for slot in 0..<t.topK {
                    expected += Array(repeating: Float(activation*Double(selected[token*t.topK+slot]+1)), count: t.hidden)
                }
            }
            let oracle = MLXArray(expected, [1,tokens,t.topK,t.hidden])
            XCTAssertLessThan(difference(mlp.project(input,indices,sortPrefill:false),oracle), 2e-6)
            XCTAssertLessThan(difference(mlp.project(input,indices,sortPrefill:true),oracle), 2e-6)
            XCTAssertLessThan(difference(mlp(input,indices),oracle), 2e-6)
        }
    }

}
