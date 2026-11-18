// Direct, single-process text-model admission probe. Not an HTTP/API benchmark.
import Foundation
import MLX
@_spi(SenseNovaDiagnostics) import MLXLLM
import MLXLMCommon
import MLXNN

struct ProbeCase: Decodable {
    let name: String
    let ids: [Int]
    let positions: [Int]
    let max_tokens: Int
}
struct ProbeProtocol: Decodable {
    let cases: [ProbeCase]
    let eos_ids: [Int]
    let logit_abs_gate: Float
    let logit_rms_gate: Float
}
enum ProbeError: Error { case invalid(String) }

func write(_ value: [String: Any], to url: URL) throws {
    try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(to: url)
}
func memory() -> [String: Any] {
    let s = Memory.snapshot()
    return ["active": s.activeMemory, "cache": s.cacheMemory, "peak": s.peakMemory]
}

let arguments = CommandLine.arguments
let options = Set(arguments.dropFirst(4))
let kvBits: Int? = options.contains("--kv8") ? 8 : (options.contains("--kv4") ? 4 : nil)
let measureLongPrefill = options.contains("--measure-long-prefill")
guard arguments.count >= 4,
      options.isSubset(of: ["--trace", "--float32-reference", "--check-chunked-prefill",
                            "--measure-long-prefill", "--kv8", "--kv4"]),
      !measureLongPrefill || (!options.contains("--trace") && !options.contains("--check-chunked-prefill")),
      !(options.contains("--kv8") && options.contains("--kv4")) else {
    throw ProbeError.invalid("usage: SenseNovaProbe MODEL_DIRECTORY CASES_JSON NEW_OUTPUT_DIRECTORY [--trace] [--float32-reference] [--check-chunked-prefill|--measure-long-prefill] [--kv8|--kv4]")
}
let directory = URL(fileURLWithPath: arguments[1])
let protocolURL = URL(fileURLWithPath: arguments[2])
let output = URL(fileURLWithPath: arguments[3])
guard !FileManager.default.fileExists(atPath: output.path) else { throw ProbeError.invalid("output exists") }
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let protocolData = try Data(contentsOf: protocolURL)
let protocolValue = try JSONDecoder().decode(ProbeProtocol.self, from: protocolData)
try protocolData.write(to: output.appendingPathComponent("cases.json"))
let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
guard json["source_revision"] as? String == "329a65f701ba8c48b73c30d0cbaa6f0449733673" else {
    throw ProbeError.invalid("probe requires pinned MLX release lineage")
}
let configuration = try JSONDecoder().decode(SenseNovaLoopConfiguration.self, from: data)
let model = SenseNovaLoopModel(configuration)
let index = try JSONSerialization.jsonObject(with: Data(contentsOf:
    directory.appendingPathComponent("model.safetensors.index.json"))) as! [String: Any]
let map = index["weight_map"] as! [String: String]
var weights = [String: MLXArray]()
let loadStart = Date()
for file in Set(map.values).sorted() {
    let shard = try MLX.loadArrays(url: directory.appendingPathComponent(file))
    guard Set(shard.keys) == Set(map.filter { $0.value == file }.keys),
          Set(shard.keys).isDisjoint(with: weights.keys) else {
        throw ProbeError.invalid("shard/index key mismatch")
    }
    weights.merge(shard) { first, _ in first }
}
if let q = json["quantization"] as? [String: Int], let bits = q["bits"], let group = q["group_size"] {
    guard [4, 6, 8].contains(bits), group == 64 else { throw ProbeError.invalid("unsupported quantization") }
    quantize(model: model, groupSize: group, bits: bits, filter: { path, _ in weights[path + ".scales"] != nil })
}
try model.update(parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)), verify: .all)
weights.removeAll()
if options.contains("--float32-reference") {
    model.apply { $0.dtype.isFloatingPoint ? $0.asType(.float32) : $0 }
}
eval(model)
try write(["load_seconds": Date().timeIntervalSince(loadStart), "memory": memory(),
           "diagnostic_float32_reference": options.contains("--float32-reference")],
          to: output.appendingPathComponent("load.json"))
print("SENSENOVA loaded \(memory())")
fflush(stdout)

var selectedLogits = [String: MLXArray]()
var results = [[String: Any]]()
for item in protocolValue.cases {
    guard !item.ids.isEmpty, item.ids.count + item.max_tokens <= 4096,
          item.positions == Array((item.ids.count - item.positions.count)..<item.ids.count) else {
        throw ProbeError.invalid("unsupported case geometry")
    }
    let input = MLXArray(item.ids.map(Int32.init)).reshaped(1, item.ids.count)
    let teacherStart = Date()
    let teacher: MLXArray?
    if measureLongPrefill {
        teacher = nil
    } else if options.contains("--trace") {
        let traced = model.diagnosticForward(input)
        teacher = traced.0
        let states = Dictionary(uniqueKeysWithValues: traced.1.enumerated().map { i, h in
            ("hidden_\(i)", h[0..., (h.dim(1) - 1)..<h.dim(1), 0...].asType(.float32))
        })
        try MLX.save(arrays: states, url: output.appendingPathComponent("trace-\(item.name).safetensors"))
    } else {
        teacher = model(input, cache: nil)
    }
    if let teacher {
        let selected = teacher[0..., (item.ids.count - item.positions.count)..<item.ids.count, 0...].asType(.float32)
        eval(selected)
        selectedLogits[item.name] = selected
        try MLX.save(arrays: [item.name: selected],
                     url: output.appendingPathComponent("teacher-\(item.name).safetensors"))
        print("SENSENOVA teacher \(item.name) seconds=\(Date().timeIntervalSince(teacherStart)) memory=\(memory())")
        fflush(stdout)
    }
    var cache = model.newCache(parameters: GenerateParameters(maxKVSize: 4096))
    let prefillStart = Date()
    var logits: MLXArray
    if options.contains("--check-chunked-prefill") || measureLongPrefill {
        let prefixEnd = item.ids.count - 1
        for start in stride(from: 0, to: prefixEnd, by: 64) {
            let chunk = model(input[0..., start..<min(start + 64, prefixEnd)], cache: cache)
            if measureLongPrefill {
                eval(cache)
                Memory.clearCache()
            } else {
                eval(chunk)
            }
        }
        // Match TokenIterator's ordinary path: prefill the prefix in chunks,
        // then evaluate the held-back final prompt token as a one-token step.
        logits = model(input[0..., prefixEnd..<item.ids.count], cache: cache)
        eval(logits)
        if let teacher {
            let fullLast = teacher[0, -1, 0...].asType(.float32)
            let chunkLast = logits[0, -1, 0...].asType(.float32)
            let delta = fullLast - chunkLast
            let maxAbs = abs(delta).max().item(Float.self)
            let rms = sqrt((delta * delta).mean()).item(Float.self)
            let sameTop = argMax(fullLast).item(Int.self) == argMax(chunkLast).item(Int.self)
            try MLX.save(arrays: ["full": fullLast, "chunked": chunkLast],
                         url: output.appendingPathComponent("chunk-logits-\(item.name).safetensors"))
            try write(["max_abs": maxAbs, "rms": rms, "same_top1": sameTop,
                       "cache_offsets": cache.map(\.offset), "chunk_tokens": 64],
                      to: output.appendingPathComponent("chunk-parity-\(item.name).json"))
            guard maxAbs <= protocolValue.logit_abs_gate, rms <= protocolValue.logit_rms_gate, sameTop else {
                throw ProbeError.invalid("chunked prefill differs from full forward")
            }
        }
    } else {
        logits = model(input, cache: cache)
        eval(logits)
    }
    func cacheBytes(_ layers: [KVCache]) -> Int {
        layers.reduce(0) { total, layer in
            total + layer.state.reduce(0) { $0 + $1.nbytes }
        }
    }
    let attentionOffsets = cache.compactMap { ($0 as? RotatingKVCache)?.offset }
    let beforeBytes = cacheBytes(cache.filter { $0 is RotatingKVCache })
    var prefillAttentionBytes = beforeBytes
    let recurrent = cache.compactMap { $0 as? MambaCache }.map(ObjectIdentifier.init)
    let quantizationStart = Date()
    if let kvBits {
        maybeQuantizeKVCache(cache: &cache, kvBits: kvBits)
        let quantized = cache.filter { $0 is QuantizedRotatingKVCache }
        guard quantized.count == 14, recurrent.count == 42,
              quantized.map(\.offset) == attentionOffsets,
              cache.compactMap({ $0 as? MambaCache }).map(ObjectIdentifier.init) == recurrent
        else { throw ProbeError.invalid("unexpected attention/recurrent cache conversion") }
        eval(quantized.flatMap(\.state))
        let afterBytes = cacheBytes(quantized)
        guard afterBytes > 0 && afterBytes < beforeBytes else {
            throw ProbeError.invalid("quantized attention cache did not reduce retained bytes")
        }
        prefillAttentionBytes = afterBytes
    }
    let quantizationSeconds = kvBits == nil ? 0 : Date().timeIntervalSince(quantizationStart)
    let prefillSeconds = Date().timeIntervalSince(prefillStart)
    let decodeStart = Date()
    var sampled = [Int]()
    var visible = [Int]()
    var firstDecodeLogits: MLXArray?
    var finish = "length"
    for position in 0..<item.max_tokens {
        let token = argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
        sampled.append(token)
        if protocolValue.eos_ids.contains(token) { finish = "eos"; break }
        visible.append(token)
        if position + 1 < item.max_tokens {
            logits = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
            eval(logits)
            if measureLongPrefill && position == 0 { firstDecodeLogits = logits }
        }
    }
    let decodeSeconds = Date().timeIntervalSince(decodeStart)
    if measureLongPrefill {
        var selected = ["last": logits[0, -1, 0...].asType(.float32)]
        if let firstDecodeLogits {
            selected["after_first_token"] = firstDecodeLogits[0, -1, 0...].asType(.float32)
        }
        try MLX.save(arrays: selected,
                     url: output.appendingPathComponent("decode-\(item.name).safetensors"))
    }
    let row: [String: Any] = ["case": item.name, "sampled_ids": sampled, "content_ids": visible,
        "finish": finish, "prefill_seconds": prefillSeconds, "decode_seconds": decodeSeconds,
        "decode_tokens_per_second": Double(max(0, sampled.count - 1)) / max(decodeSeconds, 1e-9),
        "attention_kv_bits": kvBits.map { $0 as Any } ?? NSNull(),
        "quantization_seconds": quantizationSeconds,
        "prefill_float_attention_bytes": beforeBytes,
        "prefill_attention_bytes": prefillAttentionBytes,
        "attention_visits": attentionOffsets.count,
        "recurrent_visits": recurrent.count,
        "decode_attention_bytes": cacheBytes(cache.filter { !($0 is MambaCache) }),
        "decode_recurrent_bytes": cacheBytes(cache.filter { $0 is MambaCache }),
        "cache_offsets": cache.map(\.offset),
        "memory": memory()]
    results.append(row)
    try write(["rows": results], to: output.appendingPathComponent("results.json"))
    print("SENSENOVA completion \(item.name) tokens=\(sampled.count) finish=\(finish) tps=\(Double(max(0, sampled.count - 1)) / max(decodeSeconds, 1e-9))")
    fflush(stdout)
}
if !selectedLogits.isEmpty {
    try MLX.save(arrays: selectedLogits, url: output.appendingPathComponent("selected-logits.safetensors"))
}
try write(["status": "complete", "rows": results.count, "memory": memory()],
          to: output.appendingPathComponent("outcome.json"))
