#!/usr/bin/env swift  // Keep the framework's synchronized source groups aligned with SwiftPM.
import Foundation

func run(_ command: String, _ arguments: [String]) throws -> Data {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = [command] + arguments
    let output = Pipe()
    process.standardOutput = output
    process.standardError = FileHandle.standardError
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw NSError(domain: "SourceMembership", code: Int(process.terminationStatus))
    }
    return data
}

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
guard FileManager.default.changeCurrentDirectoryPath(root.path) else {
    fatalError("Missing package root")
}
let description =
    try JSONSerialization.jsonObject(with: run("swift", ["package", "describe", "--type", "json"]))
    as! [String: Any]
let targets = description["targets"] as! [[String: Any]]
let cmlx = targets.first { $0["name"] as? String == "Cmlx" }!
let sources = Set(cmlx["sources"] as! [String])
let project = root.appendingPathComponent("xcode/MLX.xcodeproj/project.pbxproj")
let original = try String(contentsOf: project, encoding: .utf8)
var result = original

func trackedFiles(_ directory: String) throws -> [String] {
    let output = try run("git", ["-C", directory, "ls-files", "-z"])
    return String(decoding: output, as: UTF8.self).split(separator: "\0").map(String.init)
}

func generatedFiles(_ directory: String) -> [String] {
    let base = root.appendingPathComponent(directory)
    let enumerator = FileManager.default.enumerator(
        at: base, includingPropertiesForKeys: [.isRegularFileKey])!
    return enumerator.compactMap { entry -> String? in
        guard let url = entry as? URL,
            (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        else { return nil }
        return String(url.path.dropFirst(base.path.count + 1))
    }
}

func replaceList(_ identifier: String, _ property: String, _ files: [String]) {
    let objectStart = result.range(of: "\t\t\(identifier) /*")!.lowerBound
    let propertyStart = result.range(
        of: "\t\t\t\(property) = (\n", range: objectStart ..< result.endIndex)!
    let end = result.range(of: "\n\t\t\t);", range: propertyStart.upperBound ..< result.endIndex)!
        .upperBound
    let lines = Set(files).sorted().map { path -> String in
        let safe = path.range(of: "^[A-Za-z0-9_./]+$", options: .regularExpression) != nil
        let escaped = path.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(
            of: "\"", with: "\\\"")
        return "\t\t\t\t" + (safe ? path : "\"\(escaped)\"") + ","
    }
    result.replaceSubrange(
        propertyStart.lowerBound ..< end,
        with: "\t\t\t\(property) = (\n" + lines.joined(separator: "\n") + "\n\t\t\t);")
}

let core = try trackedFiles("Source/Cmlx/mlx")
replaceList(
    "C3AE9EA62EAAABFC000BD280", "membershipExceptions",
    core.filter { !sources.contains("mlx/" + $0) })

// SwiftPM puts distributed entry points in CmlxDistributedShim; the standalone
// Cmlx framework owns them directly and keeps the complete public C ABI.
let cabi = try trackedFiles("Source/Cmlx/mlx-c")
let frameworkOnly = Set(["mlx/c/distributed.cpp", "mlx/c/distributed_group.cpp"])
replaceList(
    "C3AE9E162EAAAB37000BD280", "membershipExceptions",
    cabi.filter { !sources.contains("mlx-c/" + $0) && !frameworkOnly.contains($0) })

let generated = generatedFiles("Source/Cmlx/mlx-generated")
replaceList(
    "C3CB32A92EB168CD0029A645", "membershipExceptions",
    generated.filter { !sources.contains("mlx-generated/" + $0) && !$0.hasSuffix(".metal") })
let headers = generatedFiles("Source/Cmlx/include-framework").filter {
    $0.hasSuffix(".h") || $0.hasSuffix(".hpp")
}
replaceList("C3CB32A82EB1677C0029A645", "publicHeaders", headers)

if CommandLine.arguments.contains("--check") {
    guard original == result else {
        FileHandle.standardError.write(
            Data("Xcode source membership is stale; run tools/update-xcode-membership.swift\n".utf8)
        )
        exit(1)
    }
} else if original != result {
    try result.write(to: project, atomically: true, encoding: .utf8)
}
print(
    "XCODE_MEMBERSHIP core=\(core.count) cabi=\(cabi.count) generated=\(generated.count) public_headers=\(headers.count)"
)
